#include <vector>
#include <memory>
#include <cmath>
#include <string>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

#include <iostream>
#include <chrono>

#include <stdlib.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor-fftw/basic_double.hpp>

typedef xt::xarray<double, xt::layout_type::row_major> arr;
typedef xt::xarray<std::complex<double>, xt::layout_type::row_major> arrcplx;

class Settings {

    public:

    Settings(
        double Bx_,
        double n0_,
        double v_,
        double t_,
        double dB0_,
        double dt_,
        int numT_,
        double initRadius_,
        double initEta_,
        double interfaceWidth_,
        double xlim_,
        int numPts_,
        arr g_,
        std::string simDir_
    ) :
        Bx(Bx_),
        n0(n0_),
        v(v_),
        t(t_),
        dB0(dB0_),
        dt(dt_),
        numT(numT_),
        initRadius(initRadius_),
        initEta(initEta_),
        interfaceWidth(interfaceWidth_),
        xlim(xlim_),
        numPts(numPts_),
        g(g_),
        simDir(simDir_)
    {
        A = Bx;
        B = dB0 - 2. * t * n0 + 3. * v * n0 * n0;
        C = -t - 3. * n0;
        D = v;

        for (int i = 0; i < g.shape()[0]; i++)
            gNormSq.push_back(g[{i, 0}] * g[{i, 0}] + g[{i, 1}] * g[{i, 1}]);

        if (numPts % 2 == 0) {
            std::cout << "WARNING: EVEN NumPTS! Added ONE!" << std::endl;
            numPts += 1;
        }
    }

    double Bx;
    double n0;
    double v;
    double t;
    double dB0;

    double dt;
    int numT;

    double initRadius;
    double initEta;
    double interfaceWidth;

    double xlim;
    int numPts;

    double A;
    double B;
    double C;
    double D;

    arr g;
    std::vector<double> gNormSq;

    std::string simDir;
};

class FFTSim {

    public:

    FFTSim(std::shared_ptr<Settings> sett_) : sett(std::move(sett_)) {
        init();
    }

    void init() {

        initGrid();
        initEtas();
        initGSqHat();
    }

    void initGrid() {

        x = xt::linspace<double>(-sett->xlim, sett->xlim, sett->numPts);
        auto[xm_, ym_] = xt::meshgrid(x, x);

        xm = xm_;
        ym = ym_;

        double dx = std::abs(x[{0}]) - std::abs(x[{1}]);

        arr kx = xt::fftw::fftfreq(x.shape()[0], dx);
        auto[kxm_, kym_] = xt::meshgrid(kx, kx);

        kxm = kxm_;
        kym = kym_;
    }

    void initEtas() {

        arr rad = xt::sqrt(xm * xm + ym * ym) - sett->initRadius;
        arr tan_ = 0.5 * (1. + xt::tanh(-3. * rad / sett->interfaceWidth));
        tan_ *= sett->initEta;

        for (int i = 0; i < sett->g.shape()[0]; i++)
            etas.push_back(arrcplx(tan_));
    }

    void initGSqHat() {

        arr ksum = kxm * kxm + kym * kym;

        for (int i = 0; i < sett->g.shape()[0]; i++) {

            arr val = arr(ksum);
            val += 2. * sett->g[{i, 0}] * kxm;
            val += 2. * sett->g[{i, 1}] * kym;
            val *= val;

            gSqHat.push_back(val);
        }
    }

    arrcplx ampAbsSqSum(int eta_i) {

        int is_eta_i;

        arrcplx sum_ = xt::zeros<std::complex<double>>(etas[0].shape());
        for (int eta_j = 0; eta_j < etas.size(); eta_j++) {

            is_eta_i = (int)(eta_i == eta_j);
            sum_ += (2. - is_eta_i) * etas[eta_j] * xt::conj(etas[eta_j]);
        }

        return sum_;
    }

    arrcplx nHat(int eta_i) {

        arrcplx n = xt::ones<std::complex<double>>(etas[0].shape());

        for (int eta_j = 0; eta_j < etas.size(); eta_j++) {
            if (eta_i == eta_j)
                continue;

            n *= xt::conj(etas[eta_j]);
        }

        n *= 2. * sett->C;
        n += 3. * sett->D * ampAbsSqSum(eta_i) * etas[eta_i];

        arrcplx nn = xt::fftw::fft2(n);

        return xt::eval(-1. * sett->gNormSq[eta_i] * nn);
    }

    arrcplx lagrHat(int eta_i) {
        arrcplx lagr = sett->A * gSqHat[eta_i] + sett->B;
        return xt::eval(-1. * lagr * sett->gNormSq[eta_i]);
    }

    arrcplx etaRoutine(int eta_i) {

        arrcplx lagr = lagrHat(eta_i);
        arrcplx n = nHat(eta_i);

        arrcplx exp_lagr = xt::exp(lagr * sett->dt);
        arrcplx fftEta = xt::fftw::fft2(etas[eta_i]);

        arrcplx n_eta = exp_lagr * fftEta;
        n_eta += ((exp_lagr - 1.) / lagr) * n;

        arrcplx inEta = xt::fftw::ifft2(n_eta);

        return inEta;
    }

    void runOneStep() {

        std::vector<arrcplx> n_etas;
        for (int eta_i = 0; eta_i < etas.size(); eta_i++)
            n_etas.push_back(etaRoutine(eta_i));

        etas = std::vector<arrcplx>(n_etas);
    }

    void writeConfig(std::string outDir) {

        std::ofstream outFile;
        outFile.open(outDir + "/" + "config.json");

        outFile << "{\n";
        outFile << "\t\"Bx\": " << sett->Bx << ",\n";
        outFile << "\t\"n0\": " << sett->n0 << ",\n";
        outFile << "\t\"v\": " << sett->v << ",\n";
        outFile << "\t\"t\": " << sett->t << ",\n";
        outFile << "\t\"dB0\": " << sett->dB0 << ",\n";
        outFile << "\t\"numPts\": " << sett->numPts << ",\n";
        outFile << "\t\"xlim\": " << sett->xlim << ",\n";
        outFile << "\t\"dt\": " << sett->dt << ",\n";
        outFile << "\t\"initRadius\": " << sett->initRadius << ",\n";
        outFile << "\t\"initEta\": " << sett->initEta << ",\n";
        outFile << "\t\"interfaceWidth\": " << sett->interfaceWidth << ",\n";
        outFile << "\t\"A\": " << sett->A << ",\n";
        outFile << "\t\"B\": " << sett->B << ",\n";
        outFile << "\t\"C\": " << sett->C << ",\n";
        outFile << "\t\"D\": " << sett->D << ",\n";
        outFile << "\t\"G\": [\n";

        int gx = sett->g.shape()[0];
        int gy = sett->g.shape()[1];

        for (int i = 0; i < gx; i++) {
            outFile << "\t\t[";
            for (int j = 0; j < gy-1; j++)
                outFile << sett->g[{i, j}] << ",";
            outFile << sett->g[{i, gy-1}] << "]";
            if (i != gx-1)
                outFile << ",";
            outFile << "\n";
        }
        outFile << "\t]\n";

        outFile << "}" << std::endl;
    }

    void writeStateOneEta(std::string outDir, int eta_i) {

        std::ofstream outFile;
        outFile.open(outDir + "/out_" + std::to_string(eta_i) + ".txt", std::ios::app);

        for (int i = 0; i < etas[eta_i].shape()[0] - 1; i++) {
            for (int j = 0; j < etas[eta_i].shape()[1]; j++) {
                outFile << xt::real(etas[eta_i][{i, j}]) << ",";
            }
        }

        int i = etas[eta_i].shape()[0] - 1;
        for (int j = 0; j < etas[eta_i].shape()[1] - 1; j++) {
            outFile << xt::real(etas[eta_i][{i, j}]) << ",";
        }

        int j = etas[eta_i].shape()[1] - 1;
        outFile << xt::real(etas[eta_i][{i, j}]);
        outFile << std::endl;

        outFile.close();
    }

    void resetEtaStateFiles(std::string outDir) {

        for (int eta_i = 0; eta_i < etas.size(); eta_i++) {
            std::ofstream outFile;
            outFile.open(outDir + "/out_" + std::to_string(eta_i) + ".txt");
            outFile.close();
        }
    }

    void writeState(std::string outDir) {

        for (int eta_i = 0; eta_i < etas.size(); eta_i++)
            writeStateOneEta(outDir, eta_i);
    }

    void run(std::string outDir, int write_every_i) {

        writeConfig(outDir);
        resetEtaStateFiles(outDir);

        for (size_t step = 0; step < sett->numT; step++) {

            runOneStep();

            if (step % write_every_i == 0) {
                writeState(outDir);

                float perc = (float)step / sett->numT * 100.;

                std::stringstream sstream;
                sstream << std::fixed << std::setprecision(2) << perc;

                std::cout << "Progress: " << sstream.str() << " % \r";
                std::cout.flush();
            }
        }

        // endline needed to get the progresss bar right
        std::cout << std::endl;
    }

    std::shared_ptr<Settings> sett;

    std::vector<arrcplx> etas;
    std::vector<arr> gSqHat;

    arr x;

    arr xm;
    arr ym;

    arr kxm;
    arr kym;
};

int main() {

    auto start = std::chrono::system_clock::now();

    arr G = {
        {-std::sqrt(3.) / 2., -0.5},
        {0., 1.},
        {std::sqrt(3.) / 2., -0.5}
    };

    double Bx = 0.98;
    double n0 = 0.;
    double v = 1. / 3.;
    double t = 1. / 2.;
    double dB0 = 0.01;//8. * t * t / (135. * v) * 1.0;

    //double etaInitVal = (t - xt::math::sqrt(t * t - 15. * v * dB0)) / 15. * v;
    double etaInitVal = 4. * t / (45. * v);

    std::cout << etaInitVal << " " << dB0 << std::endl;

    auto sett = std::make_shared<Settings>(
        /*Bx_             =*/ Bx,
        /*n0_             =*/ n0,
        /*v_              =*/ v,
        /*t_              =*/ t,
        /*dB0_            =*/ dB0,
        /*dt_             =*/ 0.5,
        /*numT_           =*/ 2500,
        /*initRadius_     =*/ 50.,
        /*initEta_        =*/ etaInitVal,
        /*interfaceWidth_ =*/ 3. * 3.14,
        /*xlim_           =*/ 250,
        /*numPts_         =*/ 1000,
        /*G               =*/ G,
        /*simDir          =*/ "/home/max/projects/apfc/data/test"
    );

    auto sim = FFTSim(sett);
    sim.run("/home/max/projects/apfc/data/test", 500);

    auto end = std::chrono::system_clock::now();

    auto timeDiff = end - start;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(timeDiff);
    auto mins = std::chrono::duration_cast<std::chrono::minutes>(timeDiff - hours);
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(timeDiff - hours - mins);

    std::cout << "Time Elapsed: " << hours.count() << ":" << mins.count() << ":" << secs.count() << std::endl;

    return 0;
}
