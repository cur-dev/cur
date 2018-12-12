# cur

* **Version:** 0.1-0
* **License:** [BSD 2-Clause](http://opensource.org/licenses/BSD-2-Clause)
* **Author:** Drew Schmidt


Low-level NVIDIA® CUDA™ interface for R. Experimental.


## Installation

<!-- To install the R package, run:

```r
install.package("cur")
``` -->

The development version is maintained on GitHub:

```r
remotes::install_github("wrathematics/cur")
```

You will need to have an installation of CUDA to build the package. You can download CUDA from the [nvidia website](https://developer.nvidia.com/cuda-downloads).

Also, R must have been compiled with `--enable-R-shlib=yes`. Otherwise, the package probably won't build. 



## Example

The amount of things you can really do with allocated memory is pretty small. You would have to take the memory and write your own CUDA kernel. But for the sake of demonstration, we will do some basic memory operations:

```r
library(cur)

a = 1:10
b = integer(10)

x = cudaMalloc(10, "int")
cudaMemcpy(x, a, 10, "int", "hosttodevice")
cudaMemset(x, 0, 4, "int")
cudaMemcpy(b, x, 10, "int", "devicetohost")

b
## [1]  0  0  0  0  5  6  7  8  9 10
```

Here's an example using `cudaMemGetInfo()`:

```r
n = memuse::howmany(mu(2, "gib"), ncol=1)[1]
n
## [1] 268435456

cudaMemGetInfo()
## $free
## [1] 16485974016
## 
## $total
## [1] 16936861696

y = cudaMalloc(n, "double")
cudaMemGetInfo()
## $free
## [1] 14338490368
## 
## $total
## [1] 16936861696
```

CUDA allocated memory is managed by the garbage collector, but you can manually free it with `cudaFree()` at any time.

```r
cudaFree(y)
cudaMemGetInfo()
## $free
## [1] 16485974016
## 
## $total
## [1] 16936861696
```

This will not destroy the R object, which itself must be removed and gc'd at some point. But this will not cause a double free:

```r
rm(x)
rm(y)
gc()
##          used (Mb) gc trigger (Mb) max used (Mb)
## Ncells 378058 20.2     750400 40.1   460000 24.6
## Vcells 741151  5.7    1308461 10.0   924718  7.1
```

We also have a quasi `nvidia-smi` clone written using the cur R API functions. Here's the output on a DGX-1:

```r
smi()
## Device  0 (428.00/16152.25 MiB)
## Device  1 (428.00/16152.25 MiB)
## Device  2 (428.00/16152.25 MiB)
## Device  3 (428.00/16152.25 MiB)
## Device  4 (428.12/16152.25 MiB)
## Device  5 (428.00/16152.25 MiB)
## Device  6 (428.06/16152.25 MiB)
## Device  7 (428.00/16152.25 MiB)
```
