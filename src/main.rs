use rand::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct CsrGraph {
    vertices: Vec<usize>,
    edges: Vec<usize>,
}

trait ParallelColorableGraph: Copy + Send + Sync {
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

fn jones_plassmann<T, W>(graph: T, rho: &Vec<W>) -> Vec<usize>
where
    T: ParallelColorableGraph,
    W: PartialOrd + Send + Sync,
{
    fn jp_get_color<T, W>(graph: T, rho: &Vec<W>, colors: &Vec<AtomicUsize>, v: usize) -> usize
    where
        T: ParallelColorableGraph,
        W: PartialOrd + Send + Sync,
    {
        let avail_colors: Vec<AtomicUsize> = (0..graph.degree(v) + 1)
            .into_par_iter()
            .map(AtomicUsize::new)
            .collect();

        graph
            .neighbors(v)
            .par_iter()
            .filter(|u| rho[**u] > rho[v])
            .for_each(|u| {
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

    fn jp_color<T, W>(
        graph: T,
        rho: &Vec<W>,
        colors: &Vec<AtomicUsize>,
        counters: &Vec<AtomicUsize>,
        v: usize,
    ) where
        T: ParallelColorableGraph,
        W: PartialOrd + Send + Sync,
    {
        colors[v].store(jp_get_color(graph, rho, colors, v), Ordering::Release);
        graph
            .neighbors(v)
            .par_iter()
            .filter(|u| rho[**u] < rho[v])
            .for_each(|u| {
                if counters[*u].fetch_sub(1, Ordering::SeqCst) == 1 {
                    jp_color(graph, rho, colors, counters, *u);
                }
            });
    }

    // Atomics in this vector are accessed with store/load with Acquire/Release ordering
    // Relaxed probably works too, but this is safer
    let colors: Vec<AtomicUsize> = (0..graph.num_vertices())
        .into_par_iter()
        .map(|i| AtomicUsize::new(i))
        .collect();

    let counters: Vec<AtomicUsize> = (0..graph.num_vertices())
        .into_par_iter()
        .map(|v| {
            AtomicUsize::new(
                graph
                    .neighbors(v)
                    .par_iter()
                    .filter(|u| rho[**u] > rho[v])
                    .count(),
            )
        })
        .collect();

    (0..graph.num_vertices())
        .into_par_iter()
        .filter(|v| {
            graph
                .neighbors(*v)
                .par_iter()
                .filter(|u| rho[**u] > rho[*v])
                .count()
                == 0
        })
        .for_each(|v| jp_color(graph, rho, &colors, &counters, v));

    // This should be compiled to a no-op
    colors.into_iter().map(AtomicUsize::into_inner).collect()
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

fn make_random_graph(n: usize, m: usize) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    let mut graph: Vec<_> = rayon::iter::repeatn(vec![], n).collect();

    for v in 0..n {
        for _ in 0..rng.gen_range(0..m) {
            let u = rng.gen_range(0..n);
            if u != v && !graph[v].contains(&u) {
                graph[v].push(u);
                graph[u].push(v);
            }
        }
    }

    graph
}

fn make_path_graph(n: usize) -> Vec<Vec<usize>> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            if n == 0 {
                vec![]
            } else if i == 0 {
                vec![1]
            } else if i == n - 1 {
                vec![i - 1]
            } else {
                vec![i - 1, i + 1]
            }
        })
        .collect()
}

fn make_path_csr_graph(n: usize) -> CsrGraph {
    if n == 0 {
        CsrGraph {
            vertices: vec![0],
            edges: vec![],
        }
    } else {
        CsrGraph {
            vertices: (0..n + 1)
                .into_par_iter()
                .map(|i| {
                    if i == 0 {
                        0
                    } else if i == n {
                        2 * i - 2
                    } else {
                        2 * i - 1
                    }
                })
                .collect(),
            edges: (0..2 * n - 2)
                .into_par_iter()
                .map(|i| {
                    if i == 0 {
                        1
                    } else if i == 2 * n - 3 {
                        n - 2
                    } else {
                        (i / 2) + ((i - 1) % 2)
                    }
                })
                .collect(),
        }
    }
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
    let cutoff = true;

    let mut rho: Vec<_> = rayon::iter::repeatn(0.0, graph.num_vertices()).collect();
    let mut to_remove: Vec<bool> = rayon::iter::repeatn(false, graph.num_vertices()).collect();

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

        let threshold = if cutoff { threshold.max(i) } else { threshold };

        degrees
            .par_iter()
            .map(|d| d.load(Ordering::Relaxed) <= threshold)
            .collect_into_vec(&mut to_remove);

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

fn make_sl_order_alt<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    let mut iters: Vec<Option<NonZeroUsize>> =
        rayon::iter::repeatn(None, graph.num_vertices()).collect();
    let mut degrees: Vec<usize> = Vec::with_capacity(graph.num_vertices());

    for i in 1.. {
        (0..graph.num_vertices())
            .into_par_iter()
            .map(|v| {
                if iters[v] == None {
                    graph
                        .neighbors(v)
                        .par_iter()
                        .filter(|u| iters[**u] == None)
                        .count()
                } else {
                    usize::MAX
                }
            })
            .collect_into_vec(&mut degrees);

        let threshold = *degrees.par_iter().min().unwrap_or(&usize::MAX);

        if threshold == usize::MAX {
            break;
        }

        iters.par_iter_mut().enumerate().for_each(|(v, r)| {
            if *r == None && degrees[v] <= threshold {
                *r = NonZeroUsize::new(i);
            }
        });
    }

    iters
        .into_par_iter()
        .map(|r| match r {
            None => 0,
            Some(v) => v.get(),
        })
        .map_init(|| rand::thread_rng(), |rng, v| v as f64 + rng.gen::<f64>())
        .collect()
}

fn make_sll_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    todo!()
}

fn test_coloring<T, W, F>(graph: T, gen_order: F, num_rounds: usize)
where
    T: ParallelColorableGraph + Copy,
    W: PartialOrd + Send + Sync,
    F: Fn(T) -> Vec<W>,
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

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn load_graph<P>(filename: P) -> io::Result<Vec<Vec<usize>>>
where
    P: AsRef<Path>,
{
    let mut graph = Vec::new();

    let lines = read_lines(filename)?;
    for line in lines {
        let line = line?;
        if line.starts_with("#") {
            continue;
        }

        let mut it = line.split("\t");
        let v = it.next().unwrap().parse::<usize>().unwrap() - 1;
        let u = it.next().unwrap().parse::<usize>().unwrap() - 1;

        if u >= graph.len() || v >= graph.len() {
            graph.resize(v.max(u) + 1, vec![]);
        }

        graph[v].push(u);
        graph[u].push(v);
    }

    Ok(graph)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // TODO: parse command line args to select graph and orderings

    /*
    let mut n: usize = 1_000;
    let mut m: usize = 10;
    let mut num_rounds = 3;

    if args.len() >= 2 {
        n = args[1].parse().unwrap();
    }
    if args.len() >= 3 {
        m = args[2].parse().unwrap();
    }
    if args.len() >= 4 {
        num_rounds = args[3].parse().unwrap();
    }

    let n = n;
    let m = m;
    let num_rounds = num_rounds;

    println!("n = {0} m = {1} num_rounds = {2}", n, m, num_rounds);
    */

    let num_rounds = 3;

    println!("loading graph...");
    let graph = load_graph(&args[1])?;

    //println!("building graph...");
    //let graph = make_random_graph(n, m);
    //let graph = make_path_csr_graph(1_000_000);

    // Graph statistics
    {
        let n = (&graph).num_vertices();
        let degree_sum: usize = (0..n).into_par_iter().map(|i| (&graph).degree(i)).sum();
        let degree_min: usize = (0..n)
            .into_par_iter()
            .map(|i| (&graph).degree(i))
            .min()
            .unwrap();

        println!("edges: {}", degree_sum / 2);
        println!("min degree: {}", degree_min);
        println!("max degree: {}", (&graph).max_degree());
        println!("average degree: {}", degree_sum as f64 / n as f64);
    }

    println!();
    println!("random ordering:");
    test_coloring(&graph, make_random_order, num_rounds);

    println!();
    println!("largest degree first ordering:");
    test_coloring(&graph, make_lf_order, num_rounds);

    println!();
    println!("largest log degree first ordering:");
    test_coloring(&graph, make_llf_order, num_rounds);

    println!();
    println!("smallest degree last ordering:");
    test_coloring(&graph, make_sl_order, num_rounds);

    Ok(())
}
