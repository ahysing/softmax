
### Debugging in CUDA

```shell
./compile_debug_linux.sh
cuda-gdb softmax 
set cuda break_on_launch application
```

#### Show all CUDA devices in the system runtime

```shell
cuda-gdb bikeanalysis
rbreak .*softMax.*
(cuda-gdb) info cuda devices
```

#### List all functions by a regular expression

gdb supports searching function names by regular expressions. Not that the list is long because libraries are included.
```shell
cuda-gdb bikeanalysis
start
info function .*Matrix.*
```



## Debug state for Known GPUs

For the Nvida Jetson TK1 `info cuda devices` reports


```
  Dev PCI Bus/Dev ID  Name Description SM Type SMs Warps/SM Lanes/Warp Max Regs/Lane Active SMs Mask 
    0        00:00.0 GK20A       GK20A   sm_32   1       64         32           256 0x00000000
```
