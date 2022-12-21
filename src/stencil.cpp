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
#include <filesystem>

#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xcomplex.hpp>

typedef xt::xarray<double, xt::layout_type::row_major> arr;
typedef xt::xarray<std::complex<double>, xt::layout_type::row_major> arrcplx;

template <class T>
void write2DToFile(T& in, std::string outPath) {

    std::ofstream outFile;
    outFile.open(outPath);

    for (int i = 0; i < in.shape()[0]; i++) {
        for (int j = 0; j < in.shape()[1]; j++) {
            outFile << in[{i, j}] << ",";
        }
    }

    outFile << std::endl;
    outFile.close();
}


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
        double theta_,
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
        theta(theta_),
        simDir(simDir_)
    {
        A = Bx;
        B = dB0 - 2. * t * n0 + 3. * v * n0 * n0;
        C = -t - 3. * n0;
        D = v;

        if (numPts % 2 == 0) {
            std::cout << "WARNING: EVEN NumPTS! Added ONE!" << std::endl;
            numPts += 1;
        }

        for (int i = 0; i < g.shape()[0]; i++)
            gNormSq.push_back(g[{i, 0}] * g[{i, 0}] + g[{i, 1}] * g[{i, 1}]);
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

    double theta;

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
    }

    void initGrid() {

        x = xt::linspace<double>(-sett->xlim, sett->xlim, sett->numPts);
        auto[xm_, ym_] = xt::meshgrid(x, x);

        xm = xm_;
        ym = ym_;

        dx = std::abs(x[{0}]) - std::abs(x[{1}]);
    }

    void initEtas() {

        double angle = sett->theta;

        arr rad = xm * xt::math::cos(angle) - ym * xt::math::sin(angle);
        arr tan_ = 0.5 * (1. + xt::tanh(-3. * rad / sett->interfaceWidth));
        tan_ *= sett->initEta;

        for (int i = 0; i < sett->g.shape()[0]; i++) {
            etas.push_back(arrcplx(xt::view(tan_, xt::all(), xt::all())));
            arrcplx phi = xt::zeros<std::complex<double>>(tan_.shape());
            phis.push_back(phi);
        }

    }

    arrcplx laplacian(arrcplx& in ) {

        int rows = in.shape()[0];
        int cols = in.shape()[1];

        arrcplx out = xt::zeros<std::complex<double>>({rows, cols});

        int u, l, r, d;

        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {

                u = (int)(i != 0);
                d = (int)(i != rows-1);
                l = (int)(j != 0);
                r = (int)(j != cols-1);

                out[{i,j}] =  in[{i+d,j}]+in[{i-u,j}];
                out[{i,j}] += in[{i,j+r}]+in[{i,j-l}];
                out[{i,j}] -= 4. * in[{i,j}];
            }
        }

        // TODO actual BC

        out *= 1./(dx * dx);

        return out;
    }

    arrcplx gradientProject(arrcplx& in, arrcplx& proj) {
        // central difference

        int rows = in.shape()[0];
        int cols = in.shape()[1];

        arrcplx out = xt::zeros<std::complex<double>>({rows, cols});

        int u, l, r, d;

        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++) {

                u = (int)(i != 0);
                d = (int)(i != rows-1);
                l = (int)(j != 0);
                r = (int)(j != cols-1);

                out[{i,j}] =  proj[{0}] * in[{i+d,j}]-in[{i-u,j}];
                out[{i,j}] += proj[{1}] * in[{i,j+r}]-in[{i,j-l}];
            }
        }

        // TODO actual BC

        out *= 1./(2. * dx);

        return out;
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

    arrcplx getGOp(arrcplx& in, int eta_i) {

        int rows = xm.shape()[0];
        int cols = xm.shape()[1];

        arrcplx out = xt::zeros<double>({rows, cols});

        arrcplx lapl = laplacian(in);

        arrcplx gRow = xt::row(sett->g, eta_i);
        gRow *= 2. * std::complex(0., 1.);

        arrcplx grad = gradientProject(
            in,
            gRow
        );

        out = lapl + grad;

        return out;
    }

    arrcplx getGamma(int eta_i) {

        int rows = xm.shape()[0];
        int cols = xm.shape()[1];

        arrcplx out = xt::ones<std::complex<double>>({rows, cols});

        for (int eta_j = 0; eta_j < etas.size(); eta_j++) {
            if (eta_i == eta_j)
                continue;
            out *= xt::conj(etas[eta_j]);
        }

        arrcplx absSqSum = ampAbsSqSum(eta_i);

        out *= 2. * sett->C;
        out += 3. * sett->D * absSqSum * etas[eta_i];

        return out;
    }

    void runPhiStep() {

        std::vector<arrcplx> nPhis;

        for (int eta_i = 0; eta_i < etas.size(); eta_i++) {

            arrcplx out = getGOp(etas[eta_i], eta_i);
            out *= sett->dt;
            out += phis[eta_i];

            nPhis.push_back(out);
        }

        phis = nPhis;
    }

    void runEtaStep() {

        std::vector<arrcplx> nEtas;

        for (int eta_i = 0; eta_i < etas.size(); eta_i++) {

            arrcplx gamma = getGamma(eta_i);

            arrcplx out = getGOp(phis[eta_i], eta_i);
            out += gamma;
            out *= sett->dt * sett->gNormSq[eta_i];

            out = etas[eta_i] - out;
            out *= 1. / (1. + sett->dt * sett->B * sett->gNormSq[eta_i]);

            nEtas.push_back(out);
        }

        etas = nEtas;
    }

    void runOneStep() {

        runPhiStep();
        runEtaStep();
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

        for (int i = 0; i < etas[eta_i].shape()[0]; i++) {
            for (int j = 0; j < etas[eta_i].shape()[1]; j++) {
                outFile << etas[eta_i][{i, j}] << ",";
            }
        }

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

        for (size_t step = 0; step <= sett->numT; step++) {

            runOneStep();

            if (step % write_every_i == 0) {
                writeState(outDir);

                float perc = (float)step / sett->numT * 100.;

                std::stringstream sstream;
                std::stringstream stheta;
                sstream << std::fixed << std::setprecision(2) << perc;
                stheta << std::fixed << std::setprecision(2) << sett->theta;

                std::cout << "Theta: " << stheta.str() << " Progress: " << sstream.str() << " % \r";
                std::cout.flush();
            }
        }

        // endline needed to get the progresss bar right
        std::cout << std::endl;
    }

    std::shared_ptr<Settings> sett;

    std::vector<arrcplx> etas;
    std::vector<arrcplx> phis;

    arr x;

    double dx;

    arr xm;
    arr ym;
};

void theta_run(double theta, std::string simDir) {

    arr G = {
        {-std::sqrt(3.) / 2, -0.5},
        {0., 1.},
        {std::sqrt(3.) / 2, -0.5}
    };

    auto sett = std::make_shared<Settings>(
        /*Bx_             =*/ 1.,
        /*n0_             =*/ 0.,
        /*v_              =*/ 1./3.,
        /*t_              =*/ 1./2.,
        /*dB0_            =*/ 0.04,
        /*dt_             =*/ 0.01,
        /*numT_           =*/ 1000,
        /*initRadius_     =*/ 10.,
        /*initEta_        =*/ 0.016,
        /*interfaceWidth_ =*/ 3. * 3.14,
        /*xlim_           =*/ 50.,
        /*numPts_         =*/ 201,
        /*G               =*/ G,
        /* theta          =*/ theta,
        /*simDir          =*/ simDir
    );

    auto sim = FFTSim(sett);
    sim.run(simDir, 1);
}

int main() {

    auto start = std::chrono::system_clock::now();

    auto end = std::chrono::system_clock::now();

    int thetaSize = 100;
    arr thetas = xt::linspace<double>(0., 3.14, thetaSize);

    std::string basePath = "/home/max/projects/apfc/data/stencil";

    for (int i = 0; i < thetaSize; i++) {

        std::stringstream stheta;
        stheta << std::fixed << std::setprecision(2) << thetas[{i}];

        std::stringstream path;
        path << basePath << "/" << stheta.str();
        std::filesystem::create_directory(path.str());

        theta_run(thetas[{i}], path.str());

        std::cout << i << "/" << thetaSize << " Done." << std::endl;
    }

    auto timeDiff = end - start;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(timeDiff);
    auto mins = std::chrono::duration_cast<std::chrono::minutes>(timeDiff - hours);
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(timeDiff - hours - mins);

    std::cout << "Time Elapsed: " << hours.count() << ":" << mins.count() << ":" << secs.count() << std::endl;
}

int main2() {

    arr G = {
        {-std::sqrt(3.) / 2, -0.5},
        {0., 1.},
        {std::sqrt(3.) / 2, -0.5}
    };

    auto sett = std::make_shared<Settings>(
        /*Bx_             =*/ 1.,
        /*n0_             =*/ 0.,
        /*v_              =*/ 1./3.,
        /*t_              =*/ 1./2.,
        /*dB0_            =*/ 0.044,
        /*dt_             =*/ 0.01,
        /*numT_           =*/ 100000,
        /*initRadius_     =*/ 10.,
        /*initEta_        =*/ 0.016,
        /*interfaceWidth_ =*/ 3. * 3.14,
        /*xlim_           =*/ 50.,
        /*numPts_         =*/ 201,
        /*G               =*/ G,
        /* theta          =*/ 0.,
        /*simDir          =*/ "/home/max/projects/apfc/data/sim"
    );

    auto sim = FFTSim(sett);
    //sim.run("/home/max/projects/apfc/data/sim", 10);

    write2DToFile(sim.etas[0], "/home/max/projects/apfc/tmp/stencil/eta0.txt");

    arrcplx lapl = sim.laplacian(sim.etas[0]);
    write2DToFile(lapl, "/home/max/projects/apfc/tmp/stencil/eta0_lapl.txt");

    arrcplx gr = xt::row(sett->g, 0);
    gr *= 2. * std::complex(0., 1.);

    arrcplx grad = sim.gradientProject(
        sim.etas[0],
        gr
    );
    write2DToFile(grad, "/home/max/projects/apfc/tmp/stencil/eta0_grad.txt");

    std::vector<int> a;
    a.push_back(0);
    a.push_back(1);
    a.push_back(2);

    std::cout << a[0] << " " << a[1] << std::endl;

    return 0;
}
