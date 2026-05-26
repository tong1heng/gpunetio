# 191自己怼自己

## Put BW

- client
	- LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH DOCA_GPUNETIO_LOG=7 ./examples/gpunetio_verbs_put_bw/gpunetio_verbs_put_bw -g CA:00.0 -d mlx5_0 -c 10.0.2.191 -b 1 -t 1
- server
	- LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH DOCA_GPUNETIO_LOG=7 ./examples/gpunetio_verbs_put_bw/gpunetio_verbs_put_bw -g CA:00.0 -d mlx5_0


## write latency

- client
	- LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH DOCA_GPUNETIO_LOG=7 ./examples/gpunetio_verbs_write_lat/gpunetio_verbs_write_lat -g CA:00.0 -d mlx5_0 -c 10.0.2.191
- server
	- LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH DOCA_GPUNETIO_LOG=7 ./examples/gpunetio_verbs_write_lat/gpunetio_verbs_write_lat -g CA:00.0 -d mlx5_0



# 191 client + 196 server

## Put BW

- client 191
	- LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH DOCA_GPUNETIO_LOG=7 ./examples/gpunetio_verbs_put_bw/gpunetio_verbs_put_bw -g CA:00.0 -d mlx5_0 -c 10.0.2.196
- server 196
	- LD_LIBRARY_PATH=./lib:/home/xieminhui/test_ibgda/third_party/gdrcopy-2.5.1/src:$LD_LIBRARY_PATH DOCA_GPUNETIO_LOG=7 ./examples/gpunetio_verbs_put_bw/gpunetio_verbs_put_bw -g 9D:00.0 -d mlx5_2


## write latency

- client
	- LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH DOCA_GPUNETIO_LOG=7 ./examples/gpunetio_verbs_write_lat/gpunetio_verbs_write_lat -g CA:00.0 -d mlx5_0 -c 10.0.2.196
- server
	- LD_LIBRARY_PATH=./lib:/home/xieminhui/test_ibgda/third_party/gdrcopy-2.5.1/src:$LD_LIBRARY_PATH DOCA_GPUNETIO_LOG=7 ./examples/gpunetio_verbs_write_lat/gpunetio_verbs_write_lat -g 9D:00.0 -d mlx5_2