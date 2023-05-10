use clap::{Args, Parser, ValueEnum};
use rand::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

struct VecVecGraph<T: TryInto<usize>> {
    vv: Vec<Vec<T>>,
    n_edges: usize,
}

impl<T: TryInto<usize> + PartialEq + Copy> VecVecGraph<T>
where
    <T as TryInto<usize>>::Error: std::fmt::Debug,
{
    fn add_edge(&mut self, u: T, v: T) {
        let uu = u.try_into().unwrap();
        let vv = v.try_into().unwrap();
        if uu == vv {
            panic!("Tried to add self edge");
        } else if !self.vv[uu].contains(&v) {
            self.vv[uu].push(v);
            self.vv[vv].push(u);
            self.n_edges += 1;
        }
    }
}

struct CsrGraph<T: TryInto<usize>> {
    vertices: Vec<T>,
    edges: Vec<T>,
}

impl<T> CsrGraph<T>
where
    T: TryInto<usize> + TryFrom<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    fn from_vec_vec<U>(graph: VecVecGraph<U>) -> Self
    where
        U: TryInto<T> + TryInto<usize>,
        <U as TryInto<T>>::Error: std::fmt::Debug,
    {
        let mut vertices: Vec<T> = Vec::with_capacity(graph.vv.len() + 1);
        let mut edges: Vec<T> = Vec::with_capacity(graph.n_edges * 2);

        for v in graph.vv.into_iter() {
            vertices.push(edges.len().try_into().unwrap());
            v.into_iter()
                .for_each(|v| edges.push(v.try_into().unwrap()));
        }

        vertices.push(edges.len().try_into().unwrap());
        Self { vertices, edges }
    }
}

trait SerialColorableGraph: Copy {
    type T: Copy + Send + Sync + TryFrom<usize> + TryInto<usize>;
    type VertexIter: std::iter::Iterator<Item = Self::T>;
    type NeighborIter: std::iter::Iterator<Item = Self::T>;

    fn num_vertices(&self) -> usize;
    fn iter_vertices(&self) -> Self::VertexIter;
    fn iter_neighbors(&self, v: Self::T) -> Self::NeighborIter;
    fn degree(&self, v: Self::T) -> usize;

    fn to_index(v: Self::T) -> usize {
        v.try_into().unwrap_or_else(|_| panic!())
    }
    fn from_index(v: usize) -> Self::T {
        v.try_into().unwrap_or_else(|_| panic!())
    }
}

impl<'a> SerialColorableGraph for &'a VecVecGraph<usize> {
    type T = usize;
    type VertexIter = std::ops::Range<usize>;
    type NeighborIter = std::iter::Copied<std::slice::Iter<'a, usize>>;

    fn num_vertices(&self) -> usize {
        self.vv.len()
    }

    fn iter_vertices(&self) -> Self::VertexIter {
        (0..self.vv.len()).into_iter()
    }

    fn iter_neighbors(&self, v: usize) -> Self::NeighborIter {
        self.vv[v].iter().copied()
    }

    fn degree(&self, v: usize) -> usize {
        self.vv[v].len()
    }
}

impl<'a> SerialColorableGraph for &'a CsrGraph<u32> {
    type T = u32;
    type VertexIter = std::ops::Range<u32>;
    type NeighborIter = std::iter::Copied<std::slice::Iter<'a, u32>>;

    fn num_vertices(&self) -> usize {
        self.vertices.len() - 1
    }

    fn iter_vertices(&self) -> Self::VertexIter {
        (0..u32::try_from(self.vertices.len() - 1).unwrap()).into_iter()
    }

    fn iter_neighbors(&self, v: u32) -> Self::NeighborIter {
        let o: usize = v.try_into().unwrap();
        let v1: usize = self.vertices[o].try_into().unwrap();
        let v2: usize = self.vertices[o + 1].try_into().unwrap();
        self.edges[v1..v2].iter().copied()
    }

    fn degree(&self, v: u32) -> usize {
        let o: usize = v.try_into().unwrap();
        (self.vertices[o + 1] - self.vertices[o])
            .try_into()
            .unwrap()
    }
}

impl<'a> SerialColorableGraph for &'a CsrGraph<usize> {
    type T = usize;
    type VertexIter = std::ops::Range<usize>;
    type NeighborIter = std::iter::Copied<std::slice::Iter<'a, usize>>;

    fn num_vertices(&self) -> usize {
        self.vertices.len() - 1
    }

    fn iter_vertices(&self) -> Self::VertexIter {
        (0..self.vertices.len() - 1).into_iter()
    }

    fn iter_neighbors(&self, v: usize) -> Self::NeighborIter {
        self.edges[self.vertices[v]..self.vertices[v + 1]]
            .iter()
            .copied()
    }

    fn degree(&self, v: usize) -> usize {
        self.vertices[v + 1] - self.vertices[v]
    }
}

trait ParallelColorableGraph: SerialColorableGraph + Send + Sync {
    //type T: Copy + Send + Sync + TryFrom<usize> + TryInto<usize>;
    type VertexParIter: rayon::iter::IndexedParallelIterator<Item = Self::T>;
    type NeighborParIter: rayon::iter::IndexedParallelIterator<Item = Self::T>;

    //fn num_vertices(&self) -> usize;
    fn par_iter_vertices(&self) -> Self::VertexParIter;
    fn par_iter_neighbors(&self, v: Self::T) -> Self::NeighborParIter;
    //fn degree(&self, v: Self::T) -> usize;
    fn max_degree(&self) -> usize {
        self.par_iter_vertices()
            .map(|v| self.degree(v))
            .max()
            .unwrap_or(0)
    }

    /*
    fn to_index(v: Self::T) -> usize {
        v.try_into().unwrap_or_else(|_| panic!())
    }
    fn from_index(v: usize) -> Self::T {
        v.try_into().unwrap_or_else(|_| panic!())
    }
    */
}

impl<'a> ParallelColorableGraph for &'a VecVecGraph<usize> {
    type VertexParIter = rayon::range::Iter<usize>;
    type NeighborParIter = rayon::iter::Copied<rayon::slice::Iter<'a, usize>>;

    fn par_iter_vertices(&self) -> Self::VertexParIter {
        (0..self.vv.len()).into_par_iter()
    }

    fn par_iter_neighbors(&self, v: usize) -> Self::NeighborParIter {
        self.vv[v].par_iter().copied()
    }
}

impl<'a> ParallelColorableGraph for &'a CsrGraph<u32> {
    type VertexParIter = rayon::range::Iter<u32>;
    type NeighborParIter = rayon::iter::Copied<rayon::slice::Iter<'a, u32>>;

    fn par_iter_vertices(&self) -> Self::VertexParIter {
        (0..u32::try_from(self.vertices.len() - 1).unwrap()).into_par_iter()
    }

    fn par_iter_neighbors(&self, v: u32) -> Self::NeighborParIter {
        let o: usize = v.try_into().unwrap();
        let v1: usize = self.vertices[o].try_into().unwrap();
        let v2: usize = self.vertices[o + 1].try_into().unwrap();
        self.edges[v1..v2].par_iter().copied()
    }
}

impl<'a> ParallelColorableGraph for &'a CsrGraph<usize> {
    type VertexParIter = rayon::range::Iter<usize>;
    type NeighborParIter = rayon::iter::Copied<rayon::slice::Iter<'a, usize>>;

    fn par_iter_vertices(&self) -> Self::VertexParIter {
        (0..self.vertices.len() - 1).into_par_iter()
    }

    fn par_iter_neighbors(&self, v: usize) -> Self::NeighborParIter {
        self.edges[self.vertices[v]..self.vertices[v + 1]]
            .par_iter()
            .copied()
    }
}

fn jones_plassmann<PCG, W>(graph: PCG, rho: &Vec<W>) -> Vec<usize>
where
    PCG: ParallelColorableGraph,
    W: PartialOrd + Send + Sync,
{
    struct JPData {
        colors: Vec<AtomicUsize>,
        color_masks: Vec<AtomicUsize>,
        counters: Vec<AtomicUsize>,
    }
    fn jp_get_color<PCG, W>(graph: PCG, rho: &Vec<W>, data: &JPData, v: PCG::T) -> usize
    where
        PCG: ParallelColorableGraph,
        W: PartialOrd + Send + Sync,
    {
        let mask = data.color_masks[PCG::to_index(v)].load(Ordering::Acquire);
        if mask.count_zeros() > 0 {
            mask.trailing_ones() as usize
        } else {
            let avail_colors: Vec<AtomicUsize> = (usize::BITS as usize..graph.degree(v) + 1)
                .into_par_iter()
                .map(AtomicUsize::new)
                .collect();

            graph
                .par_iter_neighbors(v)
                .filter(|u| rho[PCG::to_index(*u)] > rho[PCG::to_index(v)])
                .for_each(|u| {
                    let color = data.colors[PCG::to_index(u)].load(Ordering::Acquire);
                    if color >= usize::BITS as usize {
                        if let Some(c) = avail_colors.get(color - usize::BITS as usize) {
                            c.store(usize::MAX, Ordering::Relaxed);
                        }
                    }
                });

            avail_colors
                .into_par_iter()
                .map(AtomicUsize::into_inner)
                .min()
                .unwrap()
        }
    }

    fn jp_color<PCG, W>(graph: PCG, rho: &Vec<W>, data: &JPData, v: PCG::T)
    where
        PCG: ParallelColorableGraph,
        W: PartialOrd + Send + Sync,
    {
        let color = jp_get_color(graph, rho, data, v);
        data.colors[PCG::to_index(v)].store(color, Ordering::Release);
        graph
            .par_iter_neighbors(v)
            .filter(|u| rho[PCG::to_index(*u)] < rho[PCG::to_index(v)])
            .for_each(|u| {
                if let Some(mask) = 1_usize.checked_shl(color as u32) {
                    data.color_masks[PCG::to_index(u)].fetch_or(mask, Ordering::SeqCst);
                }
                if data.counters[PCG::to_index(u)].fetch_sub(1, Ordering::SeqCst) == 1 {
                    jp_color(graph, rho, data, u);
                }
            });
    }

    // Atomics in this vector are accessed with store/load with Acquire/Release ordering
    // Relaxed probably works too, but this is safer
    let colors: Vec<AtomicUsize> = (0..graph.num_vertices())
        .into_par_iter()
        .map(|i| AtomicUsize::new(i))
        .collect();

    let color_masks: Vec<AtomicUsize> = (0..graph.num_vertices())
        .into_par_iter()
        .map(|_| AtomicUsize::new(0))
        .collect();

    let counters: Vec<AtomicUsize> = graph
        .par_iter_vertices()
        .map(|v| {
            AtomicUsize::new(
                graph
                    .par_iter_neighbors(v)
                    .filter(|u| rho[PCG::to_index(*u)] > rho[PCG::to_index(v)])
                    .count(),
            )
        })
        .collect();

    let data = JPData {
        colors,
        color_masks,
        counters,
    };

    graph
        .par_iter_vertices()
        .filter(|v| {
            graph
                .par_iter_neighbors(*v)
                .filter(|u| rho[PCG::to_index(*u)] > rho[PCG::to_index(*v)])
                .count()
                == 0
        })
        .for_each(|v| jp_color(graph, rho, &data, v));

    // This should be compiled to a no-op
    data.colors
        .into_iter()
        .map(AtomicUsize::into_inner)
        .collect()
}

fn color_serial<SCG, W>(graph: SCG, rho: &Vec<W>) -> Vec<usize>
where
    SCG: SerialColorableGraph,
    W: PartialOrd + Copy,
{
    let mut visited = vec![false; rho.len()];
    let mut order: Vec<_> = (0..rho.len()).collect();
    order.sort_by(|i, j| rho[*j].partial_cmp(&rho[*i]).unwrap());

    let mut colors = vec![0; rho.len()];

    for i in order.into_iter() {
        let v = SCG::from_index(i);
        let mut seen = vec![false; graph.degree(v) + 1];
        for u in graph.iter_neighbors(v) {
            let j = SCG::to_index(u);
            if visited[j] && colors[j] < seen.len() {
                seen[colors[j]] = true;
            }
        }
        let color = seen.iter().position(|b| !*b).unwrap();
        colors[i] = color;
        visited[i] = true;
    }

    colors
}

fn check_coloring<T>(graph: T, coloring: &Vec<usize>)
where
    T: ParallelColorableGraph,
{
    let check = graph.par_iter_vertices().find_map_any(|v| {
        graph
            .par_iter_neighbors(v)
            .find_any(|u| coloring[T::to_index(*u)] == coloring[T::to_index(v)])
            .map(|u| (v, u))
    });

    if let Some((v, u)) = check {
        println!(
            "Found coloring conflict: {} and {} share an edge and both given color {}",
            T::to_index(v),
            T::to_index(u),
            coloring[T::to_index(v)]
        );
        std::process::exit(0);
    }
}

fn make_random_graph<T>(n: usize, m: usize) -> VecVecGraph<T>
where
    T: Copy + Send + PartialEq + TryFrom<usize> + TryInto<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
    <T as TryInto<usize>>::Error: std::fmt::Debug,
{
    let mut rng = rand::thread_rng();

    let vv: Vec<_> = rayon::iter::repeatn(vec![], n).collect();
    let mut graph = VecVecGraph { vv, n_edges: 0 };

    for vv in 0..n {
        let v = vv.try_into().unwrap();
        for _ in 0..rng.gen_range(0..m) {
            let uu = rng.gen_range(0..n);
            let u = uu.try_into().unwrap();
            graph.add_edge(v, u);
        }
    }

    graph
}

fn make_path_graph<T>(n: usize) -> VecVecGraph<T>
where
    T: Copy + Send + PartialEq + TryFrom<usize> + TryInto<usize>,
    <T as TryFrom<usize>>::Error: std::fmt::Debug,
{
    VecVecGraph {
        vv: (0..n)
            .into_par_iter()
            .map(|i| {
                if n == 0 {
                    vec![]
                } else if i == 0 {
                    vec![1.try_into().unwrap()]
                } else if i == n - 1 {
                    vec![(i - 1).try_into().unwrap()]
                } else {
                    vec![(i - 1).try_into().unwrap(), (i + 1).try_into().unwrap()]
                }
            })
            .collect(),
        n_edges: if n == 0 { 0 } else { n - 1 },
    }
}

fn make_random_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    (0..graph.num_vertices())
        .into_par_iter()
        .map_init(|| rand::thread_rng(), |rng, _| rng.gen())
        .collect()
}

fn make_lf_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    graph
        .par_iter_vertices()
        .map_init(
            || rand::thread_rng(),
            |rng, v| graph.degree(v) as f64 + rng.gen::<f64>(),
        )
        .collect()
}

fn make_llf_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    graph
        .par_iter_vertices()
        .map_init(
            || rand::thread_rng(),
            |rng, v| (graph.degree(v) as f64).log2().ceil() + rng.gen::<f64>(),
        )
        .collect()
}

fn make_sl_order<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    let cutoff = true;
    let rounds = 10;

    let mut rho: Vec<_> = rayon::iter::repeatn(0.0, graph.num_vertices()).collect();
    let mut to_remove: Vec<bool> = rayon::iter::repeatn(false, graph.num_vertices()).collect();

    let degrees: Vec<AtomicUsize> = graph
        .par_iter_vertices()
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

        let threshold = if cutoff {
            threshold.max(i / rounds)
        } else {
            threshold
        };

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
                    graph.par_iter_neighbors(T::from_index(v)).for_each(|u| {
                        let u: usize = T::to_index(u);
                        if !to_remove[u] && degrees[u].load(Ordering::Acquire) != usize::MAX {
                            degrees[u].fetch_sub(1, Ordering::SeqCst);
                        }
                    });
                }
            },
        );
    }

    rho
}

fn _make_sl_order_alt<T: ParallelColorableGraph>(graph: T) -> Vec<f64> {
    let mut iters: Vec<Option<NonZeroUsize>> =
        rayon::iter::repeatn(None, graph.num_vertices()).collect();
    let mut degrees: Vec<usize> = Vec::with_capacity(graph.num_vertices());

    for i in 1.. {
        graph
            .par_iter_vertices()
            .map(|v| {
                if iters[T::to_index(v)] == None {
                    graph
                        .par_iter_neighbors(v)
                        .filter(|u| iters[T::to_index(*u)] == None)
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
    let rounds = 10;

    let mut i = 1;

    let mut rho: Vec<_> = rayon::iter::repeatn(0.0, graph.num_vertices()).collect();
    let mut to_remove: Vec<bool> = rayon::iter::repeatn(false, graph.num_vertices()).collect();

    let degrees: Vec<AtomicUsize> = graph
        .par_iter_vertices()
        .map(|v| AtomicUsize::new(graph.degree(v)))
        .collect();

    for d in 0..graph.max_degree().ilog2() + 1 {
        for _ in 0..rounds {
            degrees
                .par_iter()
                .map(|i| i.load(Ordering::Relaxed) <= 1_usize.wrapping_shl(d))
                .collect_into_vec(&mut to_remove);

            rho.par_iter_mut().enumerate().for_each_init(
                || rand::thread_rng(),
                |rng, (v, r)| {
                    if to_remove[v] {
                        *r = i as f64 + rng.gen::<f64>();

                        degrees[v].store(usize::MAX, Ordering::Release);
                        graph.par_iter_neighbors(T::from_index(v)).for_each(|u| {
                            let u: usize = T::to_index(u);
                            if !to_remove[u] && degrees[u].load(Ordering::Acquire) != usize::MAX {
                                degrees[u].fetch_sub(1, Ordering::SeqCst);
                            }
                        });
                    }
                },
            );

            i += 1;
        }
    }

    rho
}

fn test_coloring<T, W, F>(graph: T, gen_order: F, alg: ColorAlg, num_rounds: usize)
where
    T: ParallelColorableGraph + Copy,
    W: PartialOrd + Copy + Send + Sync,
    F: Fn(T) -> Vec<W>,
{
    let run_color = |ordering: Vec<W>| match alg {
        ColorAlg::JP => jones_plassmann(graph, &ordering),
        ColorAlg::Serial => color_serial(graph, &ordering),
    };

    let mut tot_ordering_dur = Duration::new(0, 0);
    let mut tot_coloring_dur = Duration::new(0, 0);

    let mut do_round = |t| {
        println!("creating order...");
        let ordering_start = Instant::now();
        let ordering = gen_order(graph);
        let ordering_dur = ordering_start.elapsed();
        println!("finished in {}s", ordering_dur.as_secs_f64());
        if t {
            tot_ordering_dur += ordering_dur;
        }

        println!("coloring graph...");
        let coloring_start = Instant::now();
        let coloring = run_color(ordering);
        let coloring_dur = coloring_start.elapsed();
        println!("finished in {}s", coloring_dur.as_secs_f64());
        if t {
            tot_coloring_dur += coloring_dur;
        }

        println!("checking coloring...");
        check_coloring(graph, &coloring);

        let num_colors = coloring.par_iter().max().unwrap() + 1;
        println!("found {} coloring", num_colors);
    };

    // Warmup
    println!("Warmup Round");
    do_round(false);

    for i in 0..num_rounds {
        println!("Round {}:", i + 1);
        do_round(true);
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

fn load_graph<T, P>(filename: P) -> io::Result<VecVecGraph<T>>
where
    T: Copy + std::str::FromStr + TryInto<usize>,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    <T as TryInto<usize>>::Error: std::fmt::Debug,
    P: AsRef<Path>,
{
    let mut graph_vec = Vec::new();
    let mut n_edges = 0;

    let lines = read_lines(filename)?;
    for line in lines {
        let line = line?;
        if line.starts_with("#") {
            continue;
        }

        let mut it = line.split_whitespace();
        let v = it.next().unwrap().parse::<T>().unwrap();
        let u = it.next().unwrap().parse::<T>().unwrap();

        let vv = v.try_into().unwrap();
        let uu = u.try_into().unwrap();
        if uu >= graph_vec.len() || vv >= graph_vec.len() {
            graph_vec.resize(vv.max(uu) + 1, vec![]);
        }

        graph_vec[vv].push(u);
        graph_vec[uu].push(v);

        n_edges += 1;
    }

    Ok(VecVecGraph {
        vv: graph_vec,
        n_edges,
    })
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Ordering methods to use
    #[arg(short, long)]
    order: Vec<OrderingMode>,

    #[command(flatten)]
    graph: GraphType,

    #[arg(long, default_value_t = ColorAlg::JP)]
    alg: ColorAlg,

    /// Number of vertices
    #[arg(long, default_value_t = 0)]
    vertices: usize,

    /// Average number of edges per vertex
    #[arg(long, default_value_t = 0)]
    edges: usize,

    /// Number of times to run each coloring algorithm
    #[arg(short, long, default_value_t = 3)]
    rounds: usize,
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct GraphType {
    /// Load a graph from a txt file
    #[arg(short, long, value_name = "FILE")]
    load: Option<PathBuf>,

    /// Create a random graph
    #[arg(long, requires = "vertices", requires = "edges")]
    random_graph: bool,

    /// Create a path graph
    #[arg(long, requires = "vertices")]
    path_graph: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum OrderingMode {
    Rand,
    LF,
    SL,
    LLF,
    SLL,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum ColorAlg {
    JP,
    Serial,
}

impl std::fmt::Display for ColorAlg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ColorAlg::JP => write!(f, "jp"),
            ColorAlg::Serial => write!(f, "serial"),
        }
    }
}

fn main() -> io::Result<()> {
    let args = Cli::parse();

    let graph: VecVecGraph<u32> = if let Some(path) = args.graph.load {
        println!("loading graph...");
        load_graph(path)?
    } else if args.graph.random_graph {
        println!("building random graph...");
        make_random_graph(args.vertices, args.edges)
    } else if args.graph.path_graph {
        println!("building path graph...");
        make_path_graph(args.vertices)
    } else {
        panic!()
    };

    println!("converting to csr...");
    let graph: CsrGraph<u32> = CsrGraph::from_vec_vec(graph);

    // Graph statistics
    {
        let n = (&graph).num_vertices();
        let degree_sum: usize = (&graph)
            .par_iter_vertices()
            .map(|i| (&graph).degree(i))
            .sum();
        let degree_min: usize = (&graph)
            .par_iter_vertices()
            .map(|i| (&graph).degree(i))
            .min()
            .unwrap();

        println!("edges: {}", degree_sum / 2);
        println!("min degree: {}", degree_min);
        println!("max degree: {}", (&graph).max_degree());
        println!("average degree: {}", degree_sum as f64 / n as f64);
    }

    for m in args.order.iter() {
        println!();
        println!("ordering mode: {:?}", m);
        let f = match m {
            OrderingMode::Rand => make_random_order,
            OrderingMode::LF => make_lf_order,
            OrderingMode::SL => make_sl_order,
            OrderingMode::LLF => make_llf_order,
            OrderingMode::SLL => make_sll_order,
        };

        test_coloring(&graph, f, args.alg, args.rounds);
    }

    Ok(())
}
