use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct VecVecGraph {
    vs: Vec<Vec<usize>>,
}

impl VecVecGraph {
    fn new(num_vertices: usize) -> VecVecGraph {
        VecVecGraph {
            vs: (0..num_vertices)
                .into_par_iter()
                .map(|_| Vec::new())
                .collect(),
        }
    }

    // Does not check that edge doesn't already exist
    fn add_edge(&mut self, v1: usize, v2: usize) {
        self.vs[v1].push(v2);
        self.vs[v2].push(v1);
    }
}

trait ParallelColorableGraph {
    fn num_vertices(&self) -> usize;
    fn neighbors(&self, v: usize) -> &[usize];
    fn degree(&self, v: usize) -> usize;
    fn max_degree(&self) -> usize {
        (0..self.num_vertices()).into_par_iter().max().unwrap_or(0)
    }
}

impl ParallelColorableGraph for &VecVecGraph {
    fn num_vertices(&self) -> usize {
        self.vs.len()
    }

    fn neighbors(&self, v: usize) -> &[usize] {
        &self.vs[v]
    }

    fn degree(&self, v: usize) -> usize {
        self.vs[v].len()
    }
}

fn jones_plassmann<T>(graph: T, rho: &Vec<f64>) -> Vec<usize>
where
    T: ParallelColorableGraph + Send + Sync,
{
    struct JPVertex {
        pred: Vec<usize>,
        succ: Vec<usize>,
        counter: AtomicUsize,
    }

    fn jp_get_color(vs: &Vec<JPVertex>, colors: &Vec<AtomicUsize>, v: usize) -> usize {
        let n = vs[v].pred.len();

        // Atomics are written to with Relaxed ordering
        // Works because all threads are joined before reading
        let avail_colors: Vec<AtomicUsize> =
            (0..n + 1).into_par_iter().map(AtomicUsize::new).collect();

        vs[v].pred.par_iter().for_each(|u| {
            let color = colors[*u].load(Ordering::Acquire);
            if let Some(c) = avail_colors.get(color) {
                c.store(usize::MAX, Ordering::Relaxed);
            }
        });

        avail_colors
            .into_par_iter()
            .map(AtomicUsize::into_inner)
            .min()
            .unwrap()
    }

    fn jp_color(vs: &Vec<JPVertex>, colors: &Vec<AtomicUsize>, v: usize) {
        colors[v].store(jp_get_color(vs, colors, v), Ordering::Release);
        vs[v].succ.par_iter().for_each(|u| {
            if vs[*u].counter.fetch_sub(1, Ordering::SeqCst) == 1 {
                jp_color(vs, colors, *u);
            }
        });
    }

    let vs: Vec<_> = (0..graph.num_vertices())
        .into_par_iter()
        .map(|v| {
            let mut preds = Vec::new();
            let mut succs = Vec::new();
            for u in graph.neighbors(v) {
                if rho[*u] > rho[v] {
                    preds.push(*u);
                } else {
                    succs.push(*u);
                }
            }
            JPVertex {
                counter: AtomicUsize::new(preds.len()),
                pred: preds,
                succ: succs,
            }
        })
        .collect();

    // Atomics in this vector are accessed with store/load with Acquire/Release ordering
    // Relaxed probably works too, but this is safer
    let colors: Vec<AtomicUsize> = (0..graph.num_vertices())
        .into_par_iter()
        .map(|i| AtomicUsize::new(i))
        .collect();

    (0..graph.num_vertices())
        .into_par_iter()
        .filter(|v| vs[*v].pred.len() == 0)
        .for_each(|v| jp_color(&vs, &colors, v));

    // This should theoretically be compiled to a no-op
    colors.into_iter().map(AtomicUsize::into_inner).collect()

    // ... but this is guaranteed :)
    // actually not sure how it interacts with memory reordering guarantees
    //unsafe { std::mem::transmute(colors) }
}

fn check_coloring<T>(graph: T, coloring: &Vec<usize>)
where
    T: ParallelColorableGraph + Send + Sync,
{
    let check = (0..graph.num_vertices()).into_par_iter().find_map_any(|v| {
        graph
            .neighbors(v)
            .par_iter()
            .find_any(|u| coloring[**u] == coloring[v])
            .map(|u| (v, *u))
    });

    if let Some((v, u)) = check {
        println!(
            "Found coloring conflict: {} and {} share an edge and both given color {}",
            v, u, coloring[v]
        );
        std::process::exit(0);
    }
}

fn make_random_graph(n: usize, p: f64) -> VecVecGraph {
    let mut graph = VecVecGraph::new(n);
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in i + 1..n {
            if rng.gen::<f64>() < p {
                graph.add_edge(i, j);
            }
        }
    }

    graph
}

fn make_random_order(n: usize) -> Vec<f64> {
    (0..n).into_par_iter().map(|_| rand::random()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut n: usize = 1_000_000_000;
    let mut p: f64 = 0.1;
    let mut num_rounds = 3;

    if args.len() >= 2 {
        n = args[1].parse().unwrap();
    }
    if args.len() >= 3 {
        p = args[2].parse().unwrap();
    }
    if args.len() >= 4 {
        num_rounds = args[3].parse().unwrap();
    }

    let n = n;
    let p = p;
    let num_rounds = num_rounds;

    println!("n = {0} p = {1} num_rounds = {2}", n, p, num_rounds);

    println!("building graph...");
    let graph = make_random_graph(n, p);

    println!("creating order...");
    let ordering = make_random_order(n);

    println!("coloring graph...");
    let coloring = jones_plassmann(&graph, &ordering);

    println!("checking coloring...");
    check_coloring(&graph, &coloring);

    let num_colors = coloring.par_iter().max().unwrap() + 1;
    println!("Found {} coloring", num_colors);

    // Initializing in parallel, I think? (Hope)
    //let vs: Vec<_> = (0..n).into_par_iter().collect();

    //{
    //    let warmup_start = time::Instant::now();
    //    let ans = do_reduce(&vs);
    //    let warmup_dur = warmup_start.elapsed();
    //    println!("Total sum: {}", ans);
    //    println!("Warmup round running time: {}", warmup_dur.as_secs_f64());
    //}

    //let mut total_time = time::Duration::new(0, 0);
    //for i in 0..num_rounds {
    //    let start = time::Instant::now();
    //    let _ans = do_reduce(&vs);
    //    let dur = start.elapsed();

    //    println!("Round {} running time: {}", i + 1, dur.as_secs_f64());
    //    total_time += dur;
    //}

    //println!(
    //    "Average running time: {}",
    //    total_time.as_secs_f64() / num_rounds as f64
    //);
}
