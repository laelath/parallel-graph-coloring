use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

//struct VecVecGraph {
//    vs: Vec<Vec<usize>>,
//}

/*
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
*/

struct CsrGraph {
    vertices: Vec<usize>,
    edges: Vec<usize>,
}

trait ParallelColorableGraph: Send + Sync {
    fn num_vertices(&self) -> usize;
    fn neighbors(&self, v: usize) -> &[usize];
    fn degree(&self, v: usize) -> usize {
        self.neighbors(v).len()
    }
    fn max_degree(&self) -> usize {
        (0..self.num_vertices())
            .into_par_iter()
            .map(|v| self.degree(v))
            .max()
            .unwrap_or(0)
    }
}

impl ParallelColorableGraph for &Vec<Vec<usize>> {
    fn num_vertices(&self) -> usize {
        self.len()
    }

    fn neighbors(&self, v: usize) -> &[usize] {
        &self[v]
    }
}

impl ParallelColorableGraph for &CsrGraph {
    fn num_vertices(&self) -> usize {
        self.vertices.len() - 1
    }

    fn neighbors(&self, v: usize) -> &[usize] {
        &self.edges[self.vertices[v]..self.vertices[v + 1]]
    }
}

fn jones_plassmann<T>(graph: T, rho: &Vec<f64>) -> Vec<usize>
where
    T: ParallelColorableGraph,
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

    // This should be compiled to a no-op
    colors.into_iter().map(AtomicUsize::into_inner).collect()

    // ... but this is guaranteed :)
    // actually not sure how it interacts with memory reordering guarantees
    //unsafe { std::mem::transmute(colors) }
}

fn check_coloring<T>(graph: T, coloring: &Vec<usize>)
where
    T: ParallelColorableGraph,
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

fn make_random_graph(n: usize, p: f64) -> Vec<Vec<usize>> {
    let mut graph: Vec<_> = rayon::iter::repeatn(vec![], n).collect();
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in i + 1..n {
            if rng.gen::<f64>() < p {
                graph[i].push(j);
                graph[j].push(i);
            }
        }
    }

    graph
}

fn make_random_neighborhood_graph(n: usize, p: f64) -> Vec<Vec<usize>> {
    let mut graph: Vec<_> = rayon::iter::repeatn(vec![], n).collect();
    let mut rng = rand::thread_rng();
    for i in 0..n {
        for j in i + 1..n {
            if rng.gen::<f64>() < p / ((j - i + 1) as f64).log2() {
                graph[i].push(j);
                graph[j].push(i);
            }
        }
    }

    graph
}

fn make_random_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    (0..graph.num_vertices())
        .into_par_iter()
        .map_init(|| rand::thread_rng(), |rng, _| rng.gen())
        .collect()
}

fn make_lf_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    (0..graph.num_vertices())
        .into_par_iter()
        .map_init(
            || rand::thread_rng(),
            |rng, v| graph.degree(v) as f64 + rng.gen::<f64>(),
        )
        .collect()
}

fn make_llf_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    (0..graph.num_vertices())
        .into_par_iter()
        .map_init(
            || rand::thread_rng(),
            |rng, v| (graph.degree(v) as f64).log2().ceil() + rng.gen::<f64>(),
        )
        .collect()
}

fn make_sl_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    let mut rho: Vec<f64> = rayon::iter::repeatn(0.0, graph.num_vertices()).collect();

    let degrees: Vec<AtomicUsize> = (0..graph.num_vertices())
        .into_par_iter()
        .map(|v| AtomicUsize::new(graph.degree(v)))
        .collect();
    for i in 0.. {
        let threshold = degrees
            .par_iter()
            .map(|d| d.load(Ordering::Relaxed))
            .min()
            .unwrap_or(usize::MAX);

        if threshold == usize::MAX {
            break;
        }

        // TODO: use collect_into_vec to reuse rather than reallocate
        let to_remove: Vec<bool> = degrees
            .par_iter()
            .map(|d| d.load(Ordering::Relaxed) <= threshold)
            .collect();

        rho.par_iter_mut().enumerate().for_each_init(
            || rand::thread_rng(),
            |rng, (v, r)| {
                if to_remove[v] {
                    *r = i as f64 + rng.gen::<f64>();

                    degrees[v].store(usize::MAX, Ordering::Release);
                    graph.neighbors(v).par_iter().for_each(|u| {
                        if !to_remove[*u] && degrees[*u].load(Ordering::Acquire) != usize::MAX {
                            degrees[*u].fetch_sub(1, Ordering::SeqCst);
                        }
                    });
                }
            },
        );
    }

    rho
}

fn test_coloring<T, F>(graph: T, gen_order: F, num_rounds: usize)
where
    T: ParallelColorableGraph + Copy,
    F: Fn(T) -> Vec<f64>,
{
    // Warmup
    {
        println!("Warmup Round");
        println!("creating order...");
        let ordering_start = Instant::now();
        let ordering = gen_order(graph);
        let ordering_dur = ordering_start.elapsed();
        println!("finished in {}s", ordering_dur.as_secs_f64());

        println!("coloring graph...");
        let coloring_start = Instant::now();
        let coloring = jones_plassmann(graph, &ordering);
        let coloring_dur = coloring_start.elapsed();
        println!("finished in {}s", coloring_dur.as_secs_f64());

        println!("checking coloring...");
        check_coloring(graph, &coloring);

        let num_colors = coloring.par_iter().max().unwrap() + 1;
        println!("found {} coloring", num_colors);
    }

    let mut tot_ordering_dur = Duration::new(0, 0);
    let mut tot_coloring_dur = Duration::new(0, 0);

    for i in 0..num_rounds {
        println!("Round {}:", i + 1);
        println!("creating order...");
        let ordering_start = Instant::now();
        let ordering = gen_order(graph);
        let ordering_dur = ordering_start.elapsed();
        println!("finished in {}s", ordering_dur.as_secs_f64());
        tot_ordering_dur += ordering_dur;

        println!("coloring graph...");
        let coloring_start = Instant::now();
        let coloring = jones_plassmann(graph, &ordering);
        let coloring_dur = coloring_start.elapsed();
        println!("finished in {}s", coloring_dur.as_secs_f64());
        tot_coloring_dur += coloring_dur;

        println!("checking coloring...");
        check_coloring(graph, &coloring);

        let num_colors = coloring.par_iter().max().unwrap() + 1;
        println!("found {} coloring", num_colors);
    }

    println!(
        "average ordering time: {}s",
        tot_ordering_dur.as_secs_f64() / num_rounds as f64
    );
    println!(
        "average coloring time: {}s",
        tot_coloring_dur.as_secs_f64() / num_rounds as f64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut n: usize = 1_000;
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
    let graph = make_random_neighborhood_graph(n, p);

    // Graph statistics
    {
        let degree_sum: usize = (0..n).into_par_iter().map(|i| (&graph).degree(i)).sum();

        println!("edges: {}", degree_sum / 2);
        println!("max degree: {}", (&graph).max_degree());
        println!("average degree: {}", degree_sum as f64 / n as f64);
    }

    println!("random ordering:");
    test_coloring(&graph, make_random_order, num_rounds);

    println!("largest degree first ordering:");
    test_coloring(&graph, make_lf_order, num_rounds);

    println!("largest log degree first ordering:");
    test_coloring(&graph, make_llf_order, num_rounds);

    println!("smallest degree last ordering:");
    test_coloring(&graph, make_sl_order, num_rounds);
}
