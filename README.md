# Parallel Graph Coloring
An implementation of the Jones-Plassmann parallel graph coloring algorithm in Rust.

This implementation is a demonstration of several ordering methods and for testing
performance against several graphs and has not been packaged up into a library
for general use.

## Ordering methods
Implements the Largest-Log-Degree-First and Smallest-Log-Degree-Last ordering
heuristics from https://dl.acm.org/doi/10.1145/2612669.2612697.

## Usage
Can either be run on a generated random graph or load an undirected graph in the SNAP
network dataset format.
