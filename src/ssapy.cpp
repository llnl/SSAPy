#include "ssapy.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <exception>

namespace ssapy {

    void Ellipsoid::sphereToCart(
        const double lon, const double lat, const double height,
        double& x, double& y, double& z
    ) const {
        double slat2 = sin(lat)*sin(lat);
        double N = _Req / sqrt(1-_e2*slat2);  // MG (5.84)
        // MG (5.83)
        x = (N+height)*cos(lat)*cos(lon);
        y = (N+height)*cos(lat)*sin(lon);
        z = (_1mf2*N+height)*sin(lat);
    }

    void Ellipsoid::cartToSphere(
        const double x, const double y, const double z,
        double& lon, double& lat, double& height
    ) const {
        const double r2 = x*x + y*y;
        double dz = _e2*z;
        double dz1;
        double zdz, slat, N;
        while (true) {
            // MG (5.87)
            zdz = z + dz;
            slat = zdz/sqrt(r2+zdz*zdz);
            N = _Req/sqrt(1-_e2*slat*slat);
            dz1 = N*_e2*slat;
            if (fabs(dz-dz1) < 1e-6) break; // micron precision is plenty
            dz = dz1;
        }
        zdz = z+dz;
        // MG (5.88)
        lon = atan2(y, x);
        lat = atan(zdz/sqrt(r2));
        height = sqrt(r2+zdz*zdz) - N;
    }


    double HarrisPriester::density(
        const double x, const double y, const double z,
        const double ra_Sun, const double dec_Sun
    ) const {
        const double ra_lag = 0.5235987755982988;

        // Start with height so we can early exit
        double lon, lat, height;
        _ellip.cartToSphere(x, y, z, lon, lat, height);
        height *= 1.e-3; // m -> km

        if (height < _h.front()) throw std::runtime_error("Satellite has burned up on re-entry");
        if (height > _h.back()) return 0.0;

        double cos_dec_Sun = cos(dec_Sun);
        double ex_bulge = cos_dec_Sun * cos(ra_Sun + ra_lag);
        double ey_bulge = cos_dec_Sun * sin(ra_Sun + ra_lag);
        double ez_bulge = sin(dec_Sun);

        // Bulge angle scaling factor  MG (3.104)
        double cosnpsi2 = pow(
            0.5 + 0.5*(x*ex_bulge + y*ey_bulge + z*ez_bulge)/sqrt(x*x+y*y+z*z),
            _n/2
        );

        double rho_antapex;
        double rho_apex;
        if (height == _h.back()) {
            // The scale-height arrays describe the 49 intervals between the
            // 50 tabulated altitudes. At the upper table boundary there is no
            // following interval, so use the tabulated endpoint directly.
            rho_antapex = _rho_antapex.back();
            rho_apex = _rho_apex.back();
        } else {
            auto hptr = std::upper_bound(_h.cbegin(), _h.cend(), height);
            auto idx = std::distance(_h.cbegin(), hptr);
            rho_antapex = _rho_antapex[idx-1] * exp(-(height-_h[idx-1])/_scale_antapex[idx-1]);  // MG (3.101)
            rho_apex = _rho_apex[idx-1] * exp(-(height-_h[idx-1])/_scale_apex[idx-1]);
        }
        return (rho_antapex + (rho_apex - rho_antapex)*cosnpsi2) * 1e-12;
    }

    const std::array<double, 50> HarrisPriester::_h = {  // model heights [km]
        100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
        210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
        320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0,
        520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0,
        720.0, 740.0, 760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0,1000.0
    };

    const std::array<double, 50> HarrisPriester::_rho_antapex = {  // density at bulge antipode [g/km^3]
        4.974e+05, 2.490e+04, 8.377e+03, 3.899e+03, 2.122e+03, 1.263e+03,
        8.008e+02, 5.283e+02, 3.617e+02, 2.557e+02, 1.839e+02, 1.341e+02,
        9.949e+01, 7.488e+01, 5.709e+01, 4.403e+01, 3.430e+01, 2.697e+01,
        2.139e+01, 1.708e+01, 1.099e+01, 7.214e+00, 4.824e+00, 3.274e+00,
        2.249e+00, 1.558e+00, 1.091e+00, 7.701e-01, 5.474e-01, 3.916e-01,
        2.819e-01, 2.042e-01, 1.488e-01, 1.092e-01, 8.070e-02, 6.012e-02,
        4.519e-02, 3.430e-02, 2.632e-02, 2.043e-02, 1.607e-02, 1.281e-02,
        1.036e-02, 8.496e-03, 7.069e-03, 4.680e-03, 3.200e-03, 2.210e-03,
        1.560e-03, 1.150e-03
    };

    const std::array<double, 50> HarrisPriester::_rho_apex = {  // density at bulge [g/km^3]
        4.974e+05, 2.490e+04, 8.710e+03, 4.059e+03, 2.215e+03, 1.344e+03,
        8.758e+02, 6.010e+02, 4.297e+02, 3.162e+02, 2.396e+02, 1.853e+02,
        1.455e+02, 1.157e+02, 9.308e+01, 7.555e+01, 6.182e+01, 5.095e+01,
        4.226e+01, 3.526e+01, 2.511e+01, 1.819e+01, 1.337e+01, 9.955e+00,
        7.492e+00, 5.684e+00, 4.355e+00, 3.362e+00, 2.612e+00, 2.042e+00,
        1.605e+00, 1.267e+00, 1.005e+00, 7.997e-01, 6.390e-01, 5.123e-01,
        4.121e-01, 3.325e-01, 2.691e-01, 2.185e-01, 1.779e-01, 1.452e-01,
        1.190e-01, 9.776e-02, 8.059e-02, 5.741e-02, 4.210e-02, 3.130e-02,
        2.360e-02, 1.810e-02
    };

    const std::array<double, 49> HarrisPriester::_compute_scale_antapex() {
        std::array<double, 49> out{};
        for(int idx=1; idx<50; idx++) {
            double dh = HarrisPriester::_h[idx] - HarrisPriester::_h[idx-1];
            out[idx-1] = dh/log(HarrisPriester::_rho_antapex[idx-1]/HarrisPriester::_rho_antapex[idx]);
        }
        return out;
    }

    const std::array<double, 49> HarrisPriester::_compute_scale_apex() {
        std::array<double, 49> out{};
        for(int idx=1; idx<50; idx++) {
            double dh = HarrisPriester::_h[idx] - HarrisPriester::_h[idx-1];
            out[idx-1] = dh/log(HarrisPriester::_rho_apex[idx-1]/HarrisPriester::_rho_apex[idx]);
        }
	return out;
    }

    const std::array<double, 49> HarrisPriester::_scale_antapex = HarrisPriester::_compute_scale_antapex();
    const std::array<double, 49> HarrisPriester::_scale_apex = HarrisPriester::_compute_scale_apex();


    void AccelHarmonic::accel(
        const double x, const double y, const double z,
        const int n_max, const int m_max,
        double& ax, double& ay, double& az
    ) const {
        const double rsq = x*x + y*y + z*z;
        const double Rrsqr = _Rsq/rsq;

        const double x0 = _R * x / rsq;
        const double y0 = _R * y / rsq;
        const double z0 = _R * z / rsq;

        _V(0,0) = _R / sqrt(rsq);  // MG (3.31)
        _W(0,0) = 0.0;

        _V(1,0) = z0 * _V(0,0);  // MG (3.30)
        _W(1,0) = 0.0;

        // MG (3.30) with m=0
        for (int n=2; n<=n_max+1; n++) {
            _V(n,0) = ( (2*n-1)*z0*_V(n-1,0) - (n-1)*Rrsqr*_V(n-2,0))/n;
            _W(n,0) = 0.0;
        }

        for (int m=1; m<=m_max+1; m++) {
            // MG (3.29)
            _V(m,m) = (2*m-1)*(x0*_V(m-1,m-1) - y0*_W(m-1,m-1));
            _W(m,m) = (2*m-1)*(x0*_W(m-1,m-1) + y0*_V(m-1,m-1));

            // MG (3.30) with n=m+1
            if (m<=n_max) {
                _V(m+1,m) = (2*m+1)*z0*_V(m,m);
                _W(m+1,m) = (2*m+1)*z0*_W(m,m);
            }

            // MG (3.30)
            for(int n=m+2; n<=n_max+1; n++) {
                _V(n,m) = ( (2*n-1)*z0*_V(n-1,m) - (n+m-1)*Rrsqr*_V(n-2,m) ) / (n-m);
                _W(n,m) = ( (2*n-1)*z0*_W(n-1,m) - (n+m-1)*Rrsqr*_W(n-2,m) ) / (n-m);
            }
        }

        // acceleration
        double C, S;
        ax = ay = az = 0.0;
        bool overflowed = false;
        // MG (3.33)
        for (int m=0; m<=m_max; m++) {
            if (overflowed) {
                break;
            }
            for (int n=m; n<=n_max; n++) {
                if (std::isinf(_V(n+1,1)) || std::isinf(_W(n+1,1)) ||
                    std::isinf(_V(n+1,m+1)) || std::isinf(_W(n+1,m+1)) ||
                    std::isinf(_V(n+1,m-1)) || std::isinf(_W(n+1,m-1)) ||
                    std::isinf(_V(n+1,m)) || std::isinf(_W(n+1,m))) {
                    overflowed = true;
                    break;
                }
                if (m==0) {
                    C = _CS(n,0);   // = C_n,0
                    ax -= C * _V(n+1,1);
                    ay -= C * _W(n+1,1);
                    az -= (n+1)*C * _V(n+1,0);
                } else {
                    C = _CS(n,m);   // = C_n,m
                    S = _CS(m-1,n); // = S_n,m
                    double factor = 0.5*(n-m+1)*(n-m+2); // (n-m+2)!/(n-m)!/2
                    ax += 0.5*(-C*_V(n+1,m+1) - S*_W(n+1,m+1));
                    ax += factor*(C*_V(n+1,m-1) + S*_W(n+1,m-1));
                    ay += 0.5*(-C*_W(n+1,m+1) + S*_V(n+1,m+1));
                    ay += factor*(-C*_W(n+1,m-1) + S*_V(n+1,m-1));
                    az += (n-m+1)*(-C*_V(n+1,m) - S*_W(n+1,m));
                }
            }
        }

        ax *= _GMinvRsq;
        ay *= _GMinvRsq;
        az *= _GMinvRsq;
    }
}
