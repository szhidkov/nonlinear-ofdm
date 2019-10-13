/*********************************************************

	Simulation of OFDM Receiver Based on Generalized 
	Approximate Message Passing Algorithm for Polar 
	Nonlinearity Type

	For more details on the algorithm see the paper:
	S.Zhidkov, R.Dinis “Belief Propagation Receivers for 
	Near-Optimal Detection of Nonlinearly Distorted OFDM 
	Signals,” IEEE 89th Vehicular Technology Conference 
	(VTC2019-Spring), May, 2019

	Copyright (c) 2017-2019 Sergey Zhidkov
	Licensed under the MIT license.
	See LICENSE file in the project root for details.

*********************************************************/


#define ARMA_USE_CXX11
#include <iostream>
#include <fstream> 
#include <armadillo>
#include <iostream>
#include "cfg_parser.h"

using namespace std;
using namespace arma;

// Nonlinear function f(z). Some examples:
// 0 - Soft envelope limiter (G=1.7 for M=1 or 3.4 for M=4)
// 1 - 'Chaotic' nonlinearity (for N=2048)
// 100 - Phase-only nonlinearity
// 102 - TWTA model (G0=1.4 for M=1 or 2.8 for M=4)
#define NONLINEARITY_TYPE			100

#define DEBUG_SAVE 0
#define DEBUG_PRINT 0

cx_vec nln_func(cx_vec z, double G0);
cx_vec slicer(cx_vec x, double power_scale, int qam_order);

int main(int argc, char** argv)
{
	
	cx_double j = sqrt(-1);

	// Get simulation parameters from configuration file
	ConfigFile cfg("cfg.txt");
	int N = cfg.getValueOfKey<int>("N", 8192);
	int Nu = cfg.getValueOfKey<int>("Nu", N/4);
	double G0 = cfg.getValueOfKey<double>("G0", 0.51);
	double s_scale = cfg.getValueOfKey<double>("s_scale", 1.0);
	double B = cfg.getValueOfKey<double>("B", 1.0);
	int rand_seed = cfg.getValueOfKey<int>("rand_seed", 0);
	int I_max_gamp = cfg.getValueOfKey<int>("I_max", 50);
	string EbNo_init = cfg.getValueOfKey<string>("EbNo", "3.5 3.75 4.0 4.25 4.5 4.75 5.0");
	string L_SNR_init = cfg.getValueOfKey<string>("L_SNR", "500 500 500 500 500 500 500");
	int QAM_ORDER = cfg.getValueOfKey<int>("qam_order", 4);
	bool ENABLE_MULTIPATH_CHANNEL = cfg.getValueOfKey<int>("enable_multipath_channel", false);
	int N_ch = cfg.getValueOfKey<int>("N_ch", 56);
	double t_ch = cfg.getValueOfKey<double>("t_ch", 14.0);
	bool ENABLE_OOB_FILTERING_TX = cfg.getValueOfKey<int>("enable_oob_filtering_tx", false);
	bool ENABLE_OOB_FILTERING_RX = cfg.getValueOfKey<int>("enable_oob_filtering_rx", false);
	bool enable_post_correction = cfg.getValueOfKey<bool>("post_correction", false);

	int QAM_BIT = 2;
	double POWER_SCALE = 1;
	double LEVEL2_16QAM = 2.0/sqrt(10.0);
	double LEVEL2_64QAM = 2.0/sqrt(42.0);
	double LEVEL4_64QAM = 4.0/sqrt(42.0);

	if (rand_seed == 0) {
		arma_rng::set_seed_random(); 
	}
	else {
		arma_rng::set_seed(rand_seed);
	}

	vec EbNo(EbNo_init);
	ivec L_SNR(L_SNR_init);

	ofstream clog("log.txt");

	cout << "Reading simulation parameters from configuration file:" << endl;
	cout << "N = " << N << endl;
	cout << "Nu = " << Nu << endl;
	cout << "I_max = " << I_max_gamp << endl;
	cout << "G0 = " << G0 << endl;
	cout << "s_scale = " << s_scale << endl;
	cout << "B = " << B << endl;
	cout << "rand seed = " << rand_seed << endl;
	cout << "EbNo = [ ";
	for (int i = 0; i < EbNo.n_elem; i++) {
		cout << EbNo(i) << " ";
	}
	cout << "]" << endl;
	cout << "L_SNR = [ ";
	for (int i = 0; i < L_SNR.n_elem; i++) {
		cout << L_SNR(i) << " ";
	}
	cout << "]" << endl;
	cout << "Modulation: " << QAM_ORDER << "-QAM" << endl;
	cout << "Post correction = " << enable_post_correction << endl;
	cout << "Nonlinearity type = " << NONLINEARITY_TYPE << endl;
	cout << "OOB filtering (TX) = " << ENABLE_OOB_FILTERING_TX << endl;
	cout << "OOB filtering (RX) = " << ENABLE_OOB_FILTERING_RX << endl;
	cout << "Multipath channel = " << ENABLE_MULTIPATH_CHANNEL << endl;
	if (ENABLE_MULTIPATH_CHANNEL) {
		cout << "Channel taps = " << N_ch << endl;
		cout << "Channel delay spread = " << t_ch << endl;
	}

	// Copy to log
	clog << "Reading simulation parameters from configuration file:" << endl;
	clog << "N = " << N << endl;
	clog << "Nu = " << Nu << endl;
	clog << "I_max = " << I_max_gamp << endl;
	clog << "G0 = " << G0 << endl;
	clog << "s_scale = " << s_scale << endl;
	clog << "B = " << B << endl;
	clog << "rand seed = " << rand_seed << endl;
	clog << "EbNo = [ ";
	for (int i = 0; i < EbNo.n_elem; i++) {
		clog << EbNo(i) << " ";
	}
	clog << "]" << endl;
	clog << "L_SNR = [ ";
	for (int i = 0; i < L_SNR.n_elem; i++) {
		clog << L_SNR(i) << " ";
	}
	clog << "]" << endl;
	clog << "Modulation: " << QAM_ORDER << "-QAM" << endl;
	clog << "Post correction = " << enable_post_correction << endl;
	clog << "Nonlinearity type = " << NONLINEARITY_TYPE << endl;
	clog << "OOB filtering (TX) = " << ENABLE_OOB_FILTERING_TX << endl;
	clog << "OOB filtering (RX) = " << ENABLE_OOB_FILTERING_RX << endl;
	clog << "Multipath channel = " << ENABLE_MULTIPATH_CHANNEL << endl;
	if (ENABLE_MULTIPATH_CHANNEL) {
		clog << "Channel taps = " << N_ch << endl;
		clog << "Channel delay spread = " << t_ch << endl;
	}
	
	if (QAM_ORDER == 64) {
		QAM_BIT = 6;
	}
	else if (QAM_ORDER == 16) {
		QAM_BIT = 4;
	}
	else {
		QAM_BIT = 2;
	}

	vec SNR = EbNo + 10.0 * log10(double(QAM_BIT) * double(Nu) / double(N));


	vec BER_lin = zeros<vec>(SNR.n_elem);  
	vec BER_clip = zeros<vec>(SNR.n_elem);
	vec BER_gamp = zeros<vec>(SNR.n_elem);

	vec FER_lin = zeros<vec>(SNR.n_elem);
	vec FER_clip = zeros<vec>(SNR.n_elem);
	vec FER_gamp = zeros<vec>(SNR.n_elem);


	double init_std = 1.0;

	vec C;

	if (QAM_ORDER == 64) {
		C = { -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0 };
		POWER_SCALE = sqrt(42.0);
	}
	else if (QAM_ORDER == 16) {
		C = { -3.0, -1.0, 1.0, 3.0 };
		POWER_SCALE = sqrt(10.0);
	}
	else {
		C = { -1.0, 1.0 };
		POWER_SCALE = sqrt(2.0);
	}
	
	C = C * (1.0 / POWER_SCALE);

	const uword N_z = 512;
	double Z_range;
	if (QAM_ORDER == 64) {
		Z_range = 5.0;
	}
	else if (QAM_ORDER == 16) {
		Z_range = 5.0;
	}
	else {
		Z_range = 5.0;
	}
 
	cx_vec z_all = zeros<cx_vec>(N_z * N_z);
	for (int k = 0; k < N_z; k++) {
		for (int n = 0; n < N_z; n++) {
			z_all(n*N_z + k).real(Z_range * ((double(k) / double(N_z / 2)) - 1.0));
			z_all(n*N_z + k).imag(Z_range * ((double(n) / double(N_z / 2)) - 1.0));
		}
	}


	cx_vec z_all_nln = nln_func(z_all, G0);

#if DEBUG_SAVE == 1
	vec a = linspace(0, 5.0, 1024);
	vec p = zeros<vec>(1024);
	cx_vec x = cx_vec(a%cos(p), a%sin(p));
	cx_vec y = nln_func(x, 1.0);
	y.save("y.txt", raw_ascii);
	return(0);
#endif

	for (int m_snr = 0; m_snr < SNR.n_elem; m_snr++) {

		double snr = SNR(m_snr);
		int L = int(L_SNR(m_snr));

		cout << endl << "----  EbNo = " << EbNo(m_snr) << "(dB) [" << L << " symbols]  ----" << endl;
		clog << endl << "----  EbNo = " << EbNo(m_snr) << "(dB) [" << L << " symbols]  ----" << endl;

		// Allocate channel impulse responce array
		cx_mat h = zeros<cx_mat>(N_ch, L);

		// Generate signals
		cx_vec X = zeros<cx_vec>(N*L);
		cx_vec z = zeros<cx_vec>(N*L);
		cx_vec X0 = zeros<cx_vec>(N);
		cx_vec z0 = zeros<cx_vec>(N);

		for (int l = 0; l < L; l++) {

			if (QAM_ORDER == 64) {
				X0.set_real(4.0 * sign(randu<vec>(N) - 0.5) + 2.0 * sign(randu<vec>(N) - 0.5) + sign(randu<vec>(N) - 0.5));
				X0.set_imag(4.0 * sign(randu<vec>(N) - 0.5) + 2.0 * sign(randu<vec>(N) - 0.5) + sign(randu<vec>(N) - 0.5));
			}
			else if (QAM_ORDER == 16) {
				X0.set_real(2.0 * sign(randu<vec>(N) - 0.5) + sign(randu<vec>(N) - 0.5));
				X0.set_imag(2.0 * sign(randu<vec>(N) - 0.5) + sign(randu<vec>(N) - 0.5));
			}
			else {
				X0.set_real(sign(randu<vec>(N) - 0.5));
				X0.set_imag(sign(randu<vec>(N) - 0.5));
			}
			X0 = X0 * (1.0/POWER_SCALE);

			// Set to zero unused subcarriers
			if (Nu < N) {
				X0.subvec(Nu,N-1) = zeros<cx_vec>(N-Nu);
			}

			X.subvec(N*l, N*(l + 1) - 1) = X0;
			z0 = ifft(X0)*sqrt(N);
			z.subvec(N*l, N*(l + 1) - 1) = z0;

		}

#if DEBUG_SAVE == 2
		z.save("z.txt", raw_ascii);
		return(0);
#endif

		cout << "Signal generation complete!" << endl;

		cx_vec z_nln = zeros<cx_vec>(N*L);
		z_nln = nln_func(z, G0);

		double IBO = max(abs(z_nln)) / stddev(G0*z);

		cout << "IBO = " << IBO << endl;
		clog << "IBO = " << IBO << endl;

		// Out-of-band filtering (optional)
		if (ENABLE_OOB_FILTERING_TX && (Nu < N)) {
			for (int l = 0; l < L; l++) {

				z0 = z_nln.subvec(N*l, N*(l + 1) - 1);
				X0 = fft(z0) / sqrt(N);

				// Set to zero unused subcarriers
				X0.subvec(Nu, N - 1) = zeros<cx_vec>(N - Nu);

				z0 = ifft(X0)*sqrt(N);
				z_nln.subvec(N*l, N*(l + 1) - 1) = z0;

			}
		}

#if DEBUG_SAVE == 3
		z_nln.save("z_nln.txt", raw_ascii);
		return(0);
#endif

		double CF = 20.0 * log10(max(abs(z_nln)) / stddev(z_nln));
		CF = (CF < 0.0) ? 0.0 : CF;

		cout << "PAPR = " << CF << " (dB)" << endl;
		clog << "PAPR = " << CF << " (dB)" << endl;
		

		if (ENABLE_MULTIPATH_CHANNEL) {
			// Exponentially decaying multipath channel profile 
			// (equivalent to Channel C, from Thompson, "Constant-Envelope OFDM", with N_ch=56, t_ch=14.0)
			vec h_abs = zeros<vec>(N_ch);
			for (int k = 0; k < N_ch; k++) {
				h_abs(k) = exp(-double(k) / t_ch);
			}
			h_abs = h_abs / sqrt(sum(square(h_abs)));

			for (int l = 0; l < L; l++) {
				for (int k = 0; k < N_ch; k++) {
					h(k, l).real(sqrt(N) * h_abs(k) * arma::randn());
					h(k, l).imag(sqrt(N) * h_abs(k) * arma::randn());
					//h(k, l).real((k == 0) ? (sqrt(N)) : (0.0));
					//h(k, l).imag(0.0);
				}
			}

#if DEBUG_SAVE == 4
			h.save("h.txt", raw_ascii);
			return(0);
#endif

			for (int l = 0; l < L; l++) {
				cx_vec h_zp = zeros<cx_vec>(N);
				h_zp.subvec(0, N_ch - 1) = h(span::all, l);
				cx_vec H = fft(h_zp) / sqrt(N);
				z.subvec(N * l, N * (l + 1) - 1) = ifft((fft(z.subvec(N * l, N * (l + 1) - 1)) % H) / sqrt(N)) * sqrt(N);
				z_nln.subvec(N * l, N * (l + 1) - 1) = ifft((fft(z_nln.subvec(N * l, N * (l + 1) - 1)) % H) / sqrt(N)) * sqrt(N);
			}

#if DEBUG_SAVE == 5
			z.save("z2.txt", raw_ascii);
			return(0);
#endif

		} // Multipath channel

		// Channel (AWGN)
		double Sw = stddev(z) * pow(10.0, -0.05*snr) / sqrt(2);
		double Sw_nln = stddev(z_nln) * pow(10.0, -0.05*snr) / sqrt(2);

		double Sw2 = Sw * Sw;
		double Sw2_nln = Sw_nln * Sw_nln;

		//cout << "Sw2 = " << Sw2 << endl;
		//cout << "Sw2_nln = " << Sw2_nln << endl;

		cx_vec y = z + Sw*(cx_vec(arma::size(z), fill::randn));
		cx_vec y_nln = z_nln + Sw_nln*(cx_vec(arma::size(z), fill::randn));

#if DEVUG_SAVE == 6
		y.save("y.txt", raw_ascii);
		y_nln.save("y_nln.txt", raw_ascii);
		return(0);
#endif

		// Decoding linear OFDM
		cout << endl << "Linear OFDM simulation" << endl;
		clog << endl << "Linear OFDM simulation" << endl;

		double ber = 0;
		double fer = 0;
		cx_vec R0 = zeros<cx_vec>(N);

		for (int l = 0; l < L; l++) {

			R0 = fft(y.subvec(N * l, N * (l + 1) - 1)) / sqrt(N);

			if (ENABLE_MULTIPATH_CHANNEL) {
				// Linear MMSE equalizer 
				cx_vec h_zp = zeros<cx_vec>(N);
				h_zp.subvec(0, N_ch - 1) = h(span::all, l);
				cx_vec H = fft(h_zp) / sqrt(N);
				R0 = R0 % (conj(H) / (square(abs(H)) + Sw2));
			}

#if DEBUG_SAVE == 7
			R0.save("r.txt", raw_ascii);
			return(0);
#endif

			uvec err_real = (sign(real(R0)) != sign(real(X.subvec(N*l, N*(l + 1) - 1))));
			uvec err_imag = (sign(imag(R0)) != sign(imag(X.subvec(N*l, N*(l + 1) - 1))));

			if (QAM_ORDER == 64) {
				err_real += (sign(abs(real(R0)) - LEVEL4_64QAM) != sign(abs(real(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL4_64QAM));
				err_imag += (sign(abs(imag(R0)) - LEVEL4_64QAM) != sign(abs(imag(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL4_64QAM));
				err_real += (sign(abs(abs(real(R0)) - LEVEL4_64QAM) - LEVEL2_64QAM) != sign(abs(abs(real(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL4_64QAM) - LEVEL2_64QAM));
				err_imag += (sign(abs(abs(imag(R0)) - LEVEL4_64QAM) - LEVEL2_64QAM) != sign(abs(abs(imag(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL4_64QAM) - LEVEL2_64QAM));
			}
			else if (QAM_ORDER == 16) {
				err_real += (sign(abs(real(R0)) - LEVEL2_16QAM) != sign(abs(real(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL2_16QAM));
				err_imag += (sign(abs(imag(R0)) - LEVEL2_16QAM) != sign(abs(imag(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL2_16QAM));

			}

			if (Nu < N) {
				err_real.subvec(Nu, N - 1) = zeros<uvec>(N - Nu);
				err_imag.subvec(Nu, N - 1) = zeros<uvec>(N - Nu);
			}

			ber = ber + sum(err_real) + sum(err_imag);
			if ((sum(err_real) > 0) || (sum(err_imag) > 0)) {
				fer += 1;
			}

		}

		ber = ber / double(L * QAM_BIT * Nu);
		fer = fer / double(L);

		cout << "BER (linear) = " << ber << endl; 
		cout << "FER (linear) = " << fer << endl << endl;

		clog << "BER (linear) = " << ber << endl;
		clog << "FER (linear) = " << fer << endl << endl;

		clog.flush();

		BER_lin(m_snr) = ber;
		FER_lin(m_snr) = fer;


		wall_clock timer;

		// GAMP decoder simulation
		int I_max_gamp2 = I_max_gamp;
		if ((B != 1.0) || (s_scale != 1.0)) {
			I_max_gamp2 = int(2 * I_max_gamp2);
		}

		double Sw2_nln_0 = Sw2_nln;
		double B0 = B;

		if (NONLINEARITY_TYPE == 100) {
			cout << "PO-OFDM simulation" << endl;
			clog << "PO-OFDM simulation" << endl;
		}
		else {
			cout << "Nonlinear OFDM simulation" << endl;
			clog << "Nonlinear OFDM simulation" << endl;
		}

		ber = 0;
		fer = 0;
		cx_vec y0 = zeros<cx_vec>(N);
		cx_vec x_hat = zeros<cx_vec>(N);
		cx_vec x_flat = zeros<cx_vec>(N);
		vec mu_x = init_std * ones<vec>(N);
		vec mu_r = zeros<vec>(N);
		cx_vec r_hat = zeros<cx_vec>(N);
		cx_vec s_hat = zeros<cx_vec>(N);
		vec mu_s = zeros<vec>(N);
		cx_vec p_hat = zeros<cx_vec>(N);
		vec mu_p = zeros<vec>(N);
		cx_vec z_hat = zeros<cx_vec>(N);
		vec mu_z = zeros<vec>(N);
		cx_vec ifft_x = zeros<cx_vec>(N);
		cx_vec fft_s = zeros<cx_vec>(N);
		vec p_out = zeros<vec>(N_z * N_z);
		vec P_x_re = zeros<vec>(arma::size(C));
		vec P_x_im = zeros<vec>(arma::size(C));

		cx_vec x_hat_best = zeros<cx_vec>(N);

		cx_vec y0_oob = zeros<cx_vec>(N);
		cx_vec y0_ref = zeros<cx_vec>(N);

		double min_ber;
		double E_min;

		//vec mu_x_best = zeros<vec>(N);
		//vec s_hat_best = zeros<vec>(N);

		for (int l = 0; l < L; l++) {

			// int l = l_k(ii);

			timer.tic();

			y0 = y_nln.subvec(N*l, N*(l + 1) - 1);

			// Out-of-band filtering in the receiver (optional)
			if (ENABLE_OOB_FILTERING_RX && (Nu < N)) {
				X0 = fft(y0) / sqrt(N);
				// Set to zero unused subcarriers
				X0.subvec(Nu, N - 1) = zeros<cx_vec>(N - Nu);
				y0 = ifft(X0)*sqrt(N);
			}

			if (ENABLE_MULTIPATH_CHANNEL) {
				// Linear MMSE equalizer 
				cx_vec h_zp = zeros<cx_vec>(N);
				h_zp.subvec(0, N_ch - 1) = h(span::all, l);
				cx_vec H = fft(h_zp) / sqrt(N);
				X0 = fft(y0) / sqrt(N);
				X0 = X0 % (conj(H) / (square(abs(H)) + Sw2_nln_0));
				y0 = ifft(X0) * sqrt(N);
			}

			y0_ref = y0;

			// Initialize priors
			x_hat = zeros<cx_vec>(N);
			x_flat = zeros<cx_vec>(N);
			mu_x = init_std * ones<vec>(N);
			if (Nu < N) {
				mu_x.subvec(Nu, N - 1) = 0.000001 * ones<vec>(N - Nu);
			}
			mu_r = zeros<vec>(N);
			r_hat = zeros<cx_vec>(N);
			s_hat = zeros<cx_vec>(N);
			mu_s = zeros<vec>(N);
			p_hat = zeros<cx_vec>(N);
			mu_p = zeros<vec>(N);
			z_hat = zeros<cx_vec>(N);
			mu_z = zeros<vec>(N);

			// PO-OFDM
			int N_phi = 512;
			const double PI = datum::pi;
			const double sPI = sqrt(PI);
			vec phi_z = linspace<vec>(0, 2*PI, N_phi);
			cx_vec exp_z = zeros<cx_vec>(N_phi);
			exp_z.set_real(cos(phi_z));
			exp_z.set_imag(sin(phi_z));
			double A_r, phi_r, A_p, phi_p;
			double a, I0, I2, K1, exp_K1;
			cx_double I1;
			vec b = zeros<vec>(N_phi);
			vec c = zeros<vec>(N_phi);
			vec b2 = zeros<vec>(N_phi);
			vec E0 = zeros<vec>(N_phi);
			vec E1 = zeros<vec>(N_phi);
			vec E2 = zeros<vec>(N_phi);
			vec U0 = zeros<vec>(N_phi);
			vec U1 = zeros<vec>(N_phi);
			vec U2 = zeros<vec>(N_phi);


			double min_ber = datum::inf;
			double E_min = datum::inf;
			vec E = zeros<vec>(I_max_gamp2);
			double ber_k = 0;
			int t = 0;
			bool reset_flag = true;

			Sw2_nln = s_scale * Sw2_nln_0;
			B = B0;

			//cout << endl;
			//clog << endl;

			vec sc_alp = ones<vec>(I_max_gamp2);

			for (t = 0; t < I_max_gamp2; t++) {

				// Output nodes
				ifft_x = ifft(x_hat) * sqrt(N);
				double mu0 = mean(mu_x);

				for (int k = 0; k < N; k++) {


					//mu_p(k) = B * mu0 + (1 - B) * mu_p(k);
					mu_p(k) = mu0;

					p_hat(k) = ifft_x(k) - mu_p(k) * s_hat(k);


#if (NONLINEARITY_TYPE == 100)

					// Version for PO-OFDM: semi-analytical solution: 2D integral becomes 1D
					// For more details see: S.Zhidkov and R.Dinis, "Phase-only OFDM"
					A_r = abs(y0(k));
					phi_r = arg(y0(k));
					A_p = abs(p_hat(k));
					phi_p = arg(p_hat(k));
					a = 1.0 / (2.0*mu_p(k));
					K1 = (A_p*A_p) / (2.0*mu_p(k));
					exp_K1 = exp(-K1);
					b = A_p * cos(phi_p - phi_z) / mu_p(k);
					c = exp(A_r * cos(phi_r - phi_z) / Sw2_nln);
					b2 = square(b);
					E0 = exp(b2 / (4.0*a) - K1);
					E1 = (1.0 + erf(b / (2.0*sqrt(a)))) % E0;
					U0 = 1.0 / (2.0*a) * exp_K1 + (sPI / (4.0*a*sqrt(a)) * b) % E1;
					U1 = 1.0 / (4.0*a*a)*b*exp_K1 + (sPI / (4.0*a*sqrt(a)))*(1.0 + b2 / (2.0*a)) % E1;
					U2 = 1.0 / (2.0*a*a) * (1.0 + b2 / (4.0*a))*exp_K1 + (b*sPI / (8.0*a*a*sqrt(a))) % (3.0 + b2 / (2.0*a)) % E1;
					I0 = accu(c % U0);
					I1 = accu(exp_z % c % U1);
					I2 = accu(c % U2);

					z_hat(k) = I1 / I0;
					mu_z(k) = 0.5 * (I2 - abs(I1)*abs(I1) / (I0)) / (I0);  // abs(I1) - since I1 may be <0 due to numerical errors

#else
					// General polar nonlinearity (2D integration - very slow)

					p_out = exp(-real((y0(k) - z_all_nln)%conj(y0(k) - z_all_nln)) / (Sw2_nln * 2.0) - real((z_all - p_hat(k))%conj(z_all - p_hat(k))) / ( mu_p(k) * 2.0));
					cx_double mp = accu(z_all % p_out);
					double sp = accu(p_out); 

					if (sp == 0) {
						mp = 0;
					}
					else {
						mp = mp / sp;
					}

					cx_double vp = accu((z_all - mp)%conj(z_all - mp) % p_out);
					if (sp == 0) {
						vp = 0;
					}
					else {
						vp = vp / sp;
					}

					z_hat(k) = mp;
					mu_z(k) = 0.5 * real(vp);

#endif
	

					s_hat(k) = B * (z_hat(k) - p_hat(k)) / mu_p(k) + (1 - B) * s_hat(k);
					mu_s(k) = B * (1 - mu_z(k) / mu_p(k)) / mu_p(k) + (1 - B) * mu_s(k);


				}


				fft_s = fft(s_hat) / sqrt(N);
				double mu1 = 1.0 /  abs(mean(mu_s));

				for (int n = 0; n < N; n++) {

					if (reset_flag) {
						x_flat(n) = x_hat(n);
						reset_flag = false;
					}
					else {
						x_flat(n) = B * x_hat(n) + (1 - B) * x_flat(n);
					}

					//mu_r(n) = B * mu1 + (1 - B) * mu_r(n);
					mu_r(n) = mu1;

					r_hat(n) = x_flat(n) + mu_r(n) * fft_s(n);

					if (n < Nu) {

						P_x_re = exp(-square(C - real(r_hat(n))) / (2.0 * mu_r(n)));
						P_x_re = P_x_re / (sum(P_x_re) + 1e-15);

						P_x_im = exp(-square(C - imag(r_hat(n))) / (2.0 * mu_r(n)));
						P_x_im = P_x_im / (sum(P_x_im) + 1e-15);

						// mmse g_in
						x_hat(n).real(sum(C % P_x_re));
						x_hat(n).imag(sum(C % P_x_im));

						mu_x(n) = 0.5*(sum(square(C - real(x_hat(n))) % P_x_re) + sum(square(C - imag(x_hat(n))) % P_x_im));

					}
					else {
					// Decision for unused subcarriers
						x_hat(n) = 0;
						mu_x(n) = 0.00001;
					}


				}


				cx_vec y_rec = slicer(x_hat, POWER_SCALE, QAM_ORDER);

				y_rec = ifft(y_rec)*sqrt(N);

				E(t) = accu(square(abs(nln_func(y_rec, G0) - y0)));

				uvec err_re = (sign(real(x_hat)) != sign(real(X.subvec(N*l, N*(l + 1) - 1))));
				uvec err_im = (sign(imag(x_hat)) != sign(imag(X.subvec(N*l, N*(l + 1) - 1))));

				if (QAM_ORDER == 64) {
					err_re += (sign(abs(real(x_hat)) - LEVEL4_64QAM) != sign(abs(real(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL4_64QAM));
					err_im += (sign(abs(imag(x_hat)) - LEVEL4_64QAM) != sign(abs(imag(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL4_64QAM));
					err_re += (sign(abs(abs(real(x_hat)) - LEVEL4_64QAM) - LEVEL2_64QAM) != sign(abs(abs(real(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL4_64QAM) - LEVEL2_64QAM));
					err_im += (sign(abs(abs(imag(x_hat)) - LEVEL4_64QAM) - LEVEL2_64QAM) != sign(abs(abs(imag(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL4_64QAM) - LEVEL2_64QAM));
				}
				else if (QAM_ORDER == 16) {
					err_re += (sign(abs(real(x_hat)) - LEVEL2_16QAM) != sign(abs(real(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL2_16QAM));
					err_im += (sign(abs(imag(x_hat)) - LEVEL2_16QAM) != sign(abs(imag(X.subvec(N*l, N*(l + 1) - 1))) - LEVEL2_16QAM));

				}

				if (Nu < N) {
					err_re.subvec(Nu, N - 1) = zeros<uvec>(N - Nu);
					err_im.subvec(Nu, N - 1) = zeros<uvec>(N - Nu);
				}

				ber_k = double(sum(err_re) + sum(err_im));

				if (E(t) < E_min) {
					E_min = E(t);
					min_ber = ber_k;
					x_hat_best = x_hat;
				}

#if DEBUG_PRINT == 1
				cout << E(t) << "(" << ber_k << "), ";
				cout << "BER (" << t << ") = " << ber_k / double(QAM_BIT * Nu) << endl;
				//clog << E(t) << "(" << ber_k << "), ";
				//clog << "BER (" << t << ") = " << ber_k / (2 * Nu) << endl;
#endif

				if (ber_k <= 1) {
					E_min = E(t);
					min_ber = ber_k;
					x_hat_best = x_hat;
					break;
				}

				if (t==I_max_gamp) {

					// Reset damping parameters and restart processing
					B = 1.0;
					Sw2_nln = Sw2_nln_0;
					x_hat.zeros();
					s_hat.zeros();
					mu_s.zeros();
					x_flat.zeros();
					mu_r.zeros();
					r_hat.zeros();
					p_hat.zeros();
					mu_p.zeros();
					z_hat.zeros();
					mu_z.zeros();

					mu_x.fill(init_std);

					reset_flag = true;

				}

			}

			int post_corrected = 0;
			
			// Post correction (assuming genuie CRC)
			if (enable_post_correction && (min_ber > 0) && (min_ber<3)) {

				// Step 1: single error correction 
				// If there is only a single error we can correct it by N bit-flips + CRC checks
				uvec err_re = (sign(real(x_hat_best)) != sign(real(X.subvec(N*l, N*(l + 1) - 1))));
				uvec err_im = (sign(imag(x_hat_best)) != sign(imag(X.subvec(N*l, N*(l + 1) - 1))));

				if (min_ber <= 1) {
					min_ber = 0;  // We can correct single error by N bit-flips and CRC checks
					ber_k = 0;
					post_corrected = 1;
				}

				if ((post_corrected == 0) && (QAM_ORDER == 4)) {
					// Step 2: try double errors
					// Firsly, we select the lowest reliability bit and flip it
					// And then try to flip other N bits (= N-times CRC check)
					vec E_hat_re = square(sign(real(x_hat_best)) - real(x_hat_best));
					double E_hat_max_re = E_hat_re.max();

					vec E_hat_im = square(sign(imag(x_hat_best)) - imag(x_hat_best));
					double E_hat_max_im = E_hat_im.max();

					if (E_hat_max_re > E_hat_max_im) {
						uword i_cand = E_hat_re.index_max();
						err_re(i_cand) = 0;
					}
					else {
						uword i_cand = E_hat_im.index_max();
						err_im(i_cand) = 0;
					}

					if ((sum(err_re) + sum(err_im)) <= 1) {
						min_ber = 0;  // We can correct single error by N bit-flips and CRC checks
						ber_k = 0;
						post_corrected = 2;
					}
				}


			}
			

			ber = ber + min_ber;
			fer = fer + (min_ber > 0);

			double n_t = timer.toc();


			double tmp_ber = ber / double((l + 1) * QAM_BIT * Nu);
			double tt = t + post_corrected;
			cout << "Current BER estimation (l=" << l << ") = " << tmp_ber << "/" << ber_k / double(Nu*QAM_BIT) << " (" << min_ber << "/" << tt << ") | " << n_t << " (sec.)" << endl;
			clog << "Current BER estimation (l=" << l << ") = " << tmp_ber << "/" << ber_k / double(Nu*QAM_BIT) << " (" << min_ber << "/" << tt << ") " << endl;
			clog.flush();


		}

		ber = ber / double(L*Nu*QAM_BIT);
		fer = fer / L;

		cout << "BER (nonlinear) = " << ber << endl;
		cout << "FER (nonlinear) = " << fer << endl;

		clog << "BER (nonlinear) = " << ber << endl;
		clog << "FER (nonlinear) = " << fer << endl;

		cout << endl << endl;
		clog << endl << endl;

		clog.flush();

		BER_gamp(m_snr) = ber;
		FER_gamp(m_snr) = fer;


		EbNo.save("EbNo.txt", raw_ascii);
		BER_gamp.save("BER.txt", raw_ascii);
		FER_gamp.save("FER.txt", raw_ascii);

	
	}

	cout << "Simulation has finished!" << endl;

	clog.close();
	getchar();

}



/* Memoryless nonlinearity */
cx_vec nln_func(cx_vec z, double G0)
{

#if (NONLINEARITY_TYPE == 0) 

	vec T1 = { 0,   1 };
	vec a = { 1,   0 };
	vec b = { 0,   1 };

#elif (NONLINEARITY_TYPE == 1)
	//	Good at G0 = 0.45 (N=1024/512), 2D
	vec T1 = { 0,  1, 1.25, 1.5, 1.75,  2, 2.25, 2.5, 2.75,    3 };
	vec a = { 1,  2,    2,  -2,   -2,  2,    2,  -2,   -2, -0.25 };
	vec b = { 0, -2, -3.5,   4,  3.5, -4, -4.5,   6,  6.5,  1.75 };

#elif (NONLINEARITY_TYPE == 2)
	//	Good at G0 = 0.45 (N=1024/512), 2D
	vec T1 = { 0.0000, 1.0000, 1.2500, 1.5000, 1.7500, 2.0000, 2.2500, 2.5000, 2.7500, 3.0000 };
	vec a = { 1.0000, 2.0000, 2.0000, -2.0000, -2.0000, 2.0000, 2.0000, -2.0000, -2.0000, -0.5000 };
	vec b = { 0.0000, -2.0000, -3.5000, 4.0000, 3.5000, -4.0000, -4.5000, 6.0000, 6.5000, 2.5000 };

#else

	//	Linear model (default)
	vec T1 = { 0 };
	vec a = { 1 };
	vec b = { 0 };

#endif

#if (NONLINEARITY_TYPE < 100) 
	
	vec A = G0*abs(z);
	vec phi = arg(z);
	cx_vec y = zeros<cx_vec>(z.n_elem);


	for (int k = 0; k < z.n_elem; k++) {
		if (A[k] >= T1[T1.n_elem - 1]) {
			A[k] = a[T1.n_elem - 1] * A[k] + b[T1.n_elem - 1];
		}
		else {
			for (int m = 0; m < T1.n_elem - 1; m++) {
				if ((A[k] >= T1[m]) && (A[k] < T1[m + 1])) {
					A[k] = a[m] * A[k] + b[m];
					break;
				}
			}
		}
		
		//A[k] = abs(A[k]);

		y[k].real(A[k] * cos(phi[k]));
		y[k].imag(A[k] * sin(phi[k]));

	}
	
#else

#if (NONLINEARITY_TYPE == 100)
	// Phase-only OFDM
	vec phi = arg(z);
	cx_vec y = zeros<cx_vec>(z.n_elem);

	for (int k = 0; k < z.n_elem; k++) {
		y[k].real(cos(phi[k]));
		y[k].imag(sin(phi[k]));
	}

#elif (NONLINEARITY_TYPE == 101)
	// Rapp model


#elif (NONLINEARITY_TYPE == 102)

	// Saleh TWTA model
	double S_M = 1.0;
	double PHI_M = 1.0472;  // pi/3
	vec A = G0*abs(z);
	vec phi = arg(z);
	cx_vec y = zeros<cx_vec>(z.n_elem);

	for (int k = 0; k < z.n_elem; k++) {
		phi[k] = phi[k] + 2.0 * PHI_M * (S_M*A[k]) * (S_M*A[k]) / (1.0 + (S_M*A[k])*(S_M*A[k]));
		A[k] = 2.0 * (S_M*A[k]) / (1.0 + (S_M*A[k])*(S_M*A[k]));

		y[k].real(A[k] * cos(phi[k]));
		y[k].imag(A[k] * sin(phi[k]));

	}


#elif (NONLINEARITY_TYPE == 103)

	
	vec T_phi = { 0 };
	vec a_phi = { 1 };
	vec b_phi = { 0 };

	//vec T_phi = { 0,   1.0,      1.25,     1.5,   2.0,      2.5};
	//vec a_phi = { 1,     1,        1,       1,      1,       1};
	//vec b_phi = { 0, -0.7854,   0.7854,  -1.5708,  1.5708,   0};

	// SEL with chaotic phase rotation model
	vec A = G0*abs(z);
	vec phi = arg(z);
	cx_vec y = zeros<cx_vec>(z.n_elem);

	for (int k = 0; k < z.n_elem; k++) {

		if (A[k] >= T_phi[T_phi.n_elem - 1]) {
			phi[k] = a_phi[T_phi.n_elem - 1] * phi[k] + b_phi[T_phi.n_elem - 1];
		}
		else {
			for (int m = 0; m < T_phi.n_elem - 1; m++) {
				if ((A[k] >= T_phi[m]) && (A[k] < T_phi[m + 1])) {
					phi[k] = a_phi[m] * phi[k] + b_phi[m];
					break;
				}
			}
		}

		if (A[k] > 1.0) {
			A[k] = 1.0;
		}


		y[k].real(A[k] * cos(phi[k]));
		y[k].imag(A[k] * sin(phi[k]));

	}


#endif // Rapp/Saleh/Phase-only models

#endif // All models

	return(y);

}

// QAM slicer
cx_vec slicer(cx_vec x, double power_scale, int qam_order)
{
	int Nx = int(x.n_elem);
	cx_vec y = zeros<cx_vec>(arma::size(x));

	if (qam_order == 64) {

		for (int k = 0; k < Nx; k++) {

			if ((real(x(k))*power_scale) < -6.0) {
				y(k).real(-7.0 * (1.0 / power_scale));
			}
			else if ((real(x(k))*power_scale) < -4.0) {
				y(k).real(-5.0 * (1.0 / power_scale));
			}
			else if ((real(x(k))*power_scale) < -2.0) {
				y(k).real(-3.0 * (1.0 / power_scale));
			}
			else if ((real(x(k))*power_scale) < 0.0) {
				y(k).real(-1.0 * (1.0 / power_scale));
			}
			else if ((real(x(k))*power_scale) < 2.0) {
				y(k).real(1.0 * (1.0 / power_scale));
			}
			else if ((real(x(k))*power_scale) < 4.0) {
				y(k).real(3.0 * (1.0 / power_scale));
			}
			else if ((real(x(k))*power_scale) < 6.0) {
				y(k).real(5.0 * (1.0 / power_scale));
			}
			else {
				y(k).real(7.0 * (1.0 / power_scale));
			}

			if ((imag(x(k))*power_scale) < -6.0) {
				y(k).imag(-7.0 * (1.0 / power_scale));
			}
			else if ((imag(x(k))*power_scale) < -4.0) {
				y(k).imag(-5.0 * (1.0 / power_scale));
			}
			else if ((imag(x(k))*power_scale) < -2.0) {
				y(k).imag(-3.0 * (1.0 / power_scale));
			}
			else if ((imag(x(k))*power_scale) < 0.0) {
				y(k).imag(-1.0 * (1.0 / power_scale));
			}
			else if ((imag(x(k))*power_scale) < 2.0) {
				y(k).imag(1.0 * (1.0 / power_scale));
			}
			else if ((imag(x(k))*power_scale) < 4.0) {
				y(k).imag(3.0 * (1.0 / power_scale));
			}
			else if ((imag(x(k))*power_scale) < 6.0) {
				y(k).imag(5.0 * (1.0 / power_scale));
			}
			else {
				y(k).imag(7.0 * (1.0 / power_scale));
			}


		}

	}
	else if (qam_order == 16) {

		for (int k = 0; k < Nx; k++) {
			
			if ((real(x(k))*power_scale) < -2.0) {
				y(k).real(-3.0 * (1.0 / power_scale));
			}
			else if ((real(x(k))*power_scale) < 0.0) {
				y(k).real(-1.0 * (1.0 / power_scale));
			}
			else if ((real(x(k))*power_scale) < 2.0) {
				y(k).real(1.0 * (1.0 / power_scale));
			}
			else {
				y(k).real(3.0 * (1.0 / power_scale));
			}

			if ((imag(x(k))*power_scale) < -2.0) {
				y(k).imag(-3.0 * (1.0 / power_scale));
			}
			else if ((imag(x(k))*power_scale) < 0.0) {
				y(k).imag(-1.0 * (1.0 / power_scale));
			}
			else if ((imag(x(k))*power_scale) < 2.0) {
				y(k).imag(1.0 * (1.0 / power_scale));
			}
			else {
				y(k).imag(3.0 * (1.0 / power_scale));
			}

		}

	}
	else {
		y.set_real((1.0 / power_scale) * sign(real(x)));
		y.set_imag((1.0 / power_scale) * sign(imag(x)));
	}

	return(y);

}