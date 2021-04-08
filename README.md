# FractalRocks
Extract fractal proprties from rocks. Fractal properties of pores and grains from imagery (neutron and X-Ray tomography and thin sections) by autocorrelation. Pore properties from (U)SANS.

A type of autocorrelation is used that relates individual pixels or voxels to those some *e* distance away.

![c(e) = \frac{<(g(x)-<g>)(g(x+e)-<g>)>}{<g><g>}](https://latex.codecogs.com/svg.latex?c(e)&space;=&space;\frac{<(g(x)-<g>)(g(x&plus;e)-<g>)>}{<g><g>},)
 
 where *g* is grayscale intensity.
