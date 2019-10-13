# Belief-propagation Receivers for Nonlinear OFDM 

This is a communication system simulation code for several OFDM receivers based on generalized approximate message passing algorithm. For more details, see Ref [1, 2].

## Getting Started

This simulation code relies on Armadillo (C++ library for linear algebra & scientific computing). Other than that there are no external dependencies.

### Prerequisites

Firstly, you need to install Armadillo C++ library. You can download a copy of Armadillo C++ library from the official web-site (http://arma.sourceforge.net/), or you can use pre-built Armadillo packages provided by many Linux-based operating systems.
Personally, I'm using Windows with Visual Studio 2019 and install Armadillo via vcpkg package manager (https://github.com/Microsoft/vcpkg). 

## Running the simulation

The principal way to run simulation is via command line interface with simulation parameters specified in cfg.txt file.

### Simulation parameters

The parameters that can be specified in cfg.txt file are given in the table below:


| Parameter | Description |
|--|--|
| N | the number of subcarriers in oversampled signal  |
| Nu | the number of nonzero subcarriers (i.e., oversampling factor M=N/Nu) | 
| I_max | the maximum number of iterations |
| G0 | the input signal scaling factor |
| s_scale | the noise variance scaling factor |
| B | the damping factor |
| rand_seed | the random seed |
| EbNo | the signal-to-noise ratio per bit (array, e.g. 7.75 8.0 8.25) |
| L_SNR | the number of OFDM symbols to simulate per EbNo (array) |
| qam_order | the QAM modulation order (supported: 4, 16, 64) |
| enable_multipath_channel | set this value to 1 to simulate Rayleigh block-fading channel |
| Channel taps | the number of channel taps |
| Channel delay spread | the channel RMS delay-spread (normalized) |
| enable_oob_filtering_tx | set this value to 1 to enable out-of-band filterign in the transmitter |
| enable_oob_filtering_rx | set this value to 1 to enable out-of-band filterign in the receiver |
| post_correction | set this value to 1 to enable 1-bit correction post-processing |


### Example configuration file: 

```
N = 16384
Nu = 4096
I_max = 75
G0 = 0.0
s_scale = 0.71
B = 0.875
rand_seed = 12345
EbNo =  7.75 8.0 8.25 
L_SNR = 1000 1000 1000
qam_order = 16
enable_multipath_channel = 0
enable_oob_filtering_tx = 0
enable_oob_filtering_rx = 0
post_correction = 1
```

### Getting results

The simulation results are stored in:
* EbNo.txt (all simulated EbNo values)
* BER.txt (the BER result for each EbNo value)
* FER.txt (the FER result for each EbNo value)
* log.txt (all the output you see on the screen during simulation)

## References

1. S. Zhidkov and R.Dinis “Belief Propagation Receivers for Near-Optimal Detection of Nonlinearly Distorted OFDM Signals,” 2019 IEEE 89th Vehicular Technology Conference (VTC2019-Spring), May 2019 [Download](http://www.cifrasoft.com/people/szhidkov/papers/belief-propagation-receivers-vtc2019.pdf)
2. S. Zhidkov, "Orthogonal transform multiplexing with memoryless nonlinearity: a possible alternative to traditional coded-modulation schemes,“ in Proc. 9th International Congress on Ultra Modern Telecommunications and Control Systems (ICUMT-2017), Munich, Germany, November, 2017 [Download](https://arxiv.org/pdf/1703.03141)


## Authors

* Sergey Zhidkov - *Initial work* (https://github.com/szhidkov)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


