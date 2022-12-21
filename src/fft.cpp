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

template <class T>
void printShape(T& in) {
    std::cout << in.shape()[0] << " " << in.shape()[1] << std::endl;
}

template <class T>
void printArr(T& in) {

    int rows = in.shape()[0];
    int cols = in.shape()[1];

    if (cols > 1000)
        cols = 1;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            std::cout << in[{i, j}] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <class T>
void printArr1D(T& in) {
    int rows = in.shape()[0];
    for (int i = 0; i < rows; i++) {
        std::cout << in[{i}] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

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

template <class T>
void write1DToFile(T& in, std::string outPath) {

    std::ofstream outFile;
    outFile.open(outPath);

    for (int i = 0; i < in.shape()[0]; i++) {
        outFile << in[{i}] << ",";
    }

    outFile << std::endl;
    outFile.close();
}

template <class T>
T copySwap(T& in) {

    int rows = in.shape()[0];
    int cols = in.shape()[1];

    int k = cols;
    int kDiff = rows - k;
    int kColDiff = cols - kDiff;

    T zeros = xt::ones<std::complex<double>>({rows, rows});
    //zeros = zeros * 5.;

    // broadcast initial shape to top
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            zeros[{i, j}] = in[{i, j}];
        }
    }

    //return zeros;

    /*in = {
        {0., 0., 0., 0., 0., 3.1, 3.2, 3.3},
        {0., 0., 0., 0., 0., 3.4, 3.5, 3.6},
        {0., 0., 4., 4., 4., 3.7, 3.8, 3.9},
        {0., 0., 4., 4., 4., 3.10, 3.11, 3.12},
        {0., 0., 4., 4., 4., 3.13, 3.14, 3.15}
    };*/
    /*in = {
        {0., 0., 0., 0., 0., 3., 3., 3.},
        {0., 0., 0., 0., 0., 3., 3., 3.},
        {0., 0., 4.1, 4.2, 4.3, 3., 3., 3.},
        {0., 0., 4.4, 4.5, 4.6, 3., 3., 3.},
        {0., 0., 4.7, 4.8, 4.9, 3., 3., 3.}
    };*/
    /*in = {
        {0., 0., 0., 0., 0., 3., 3., 3.},
        {0., 0., 0., 0., 0., 3., 3., 3.},
        {0., 0., 4., 4., 4., 3., 3., 3.},
        {0., 0., 4., 4., 4., 3., 3., 3.},
        {0., 0., 4., 4., 4., 3., 3., 3.}
    };*/

    //in = xt::transpose(in);

    // gett box 3 -> Done
    for (int i = k; i < rows; i++) {
        for (int j = 0; j < k; j++) {
            zeros[{j, i}] = in[{i, j}];
        }
    }

    // get box 4 -> done, but does it need the transpose?
    int isEven = (int)(rows % 2 == 0);
    for (int i = kColDiff; i < k; i++) {
        for (int j = kColDiff; j < k; j++) {
            //zeros[{j+kDiff, i+kDiff}] = in[{k-i+1, k-j+1}];
            zeros[{i+kDiff, j+kDiff}] = xt::conj(in[{k-i+isEven, k-j+isEven}]);
            //zeros[{i+kDiff, j+kDiff}] = in[{k-i+isEven, k-j+isEven}];
        }
    }

    return zeros;
}

template <class T>
arr readReal(std::string path, T shape) {

    arr out = xt::zeros<double>(shape);

    std::ifstream inFile;
    inFile.open(path, std::ios::in);

    std::string elem;
    std::stringstream lineStream;

    for (int i = 0; i < shape[0]; i++) {

        std::getline(inFile, elem, '\n');
        lineStream = std::stringstream(elem);

        for (int j = 0; j < shape[1]; j++) {
            std::getline(lineStream, elem, ',');
            out[{i, j}] = std::stod(elem);
        }
    }

    return out;
}

template <class T>
arrcplx readComplex(std::string path, T shape) {

    arrcplx out = xt::zeros<std::complex<double>>(shape);

    std::ifstream inFile;
    inFile.open(path, std::ios::in);

    std::string elem;
    std::stringstream lineStream;

    double d1;
    double d2;

    std::string cplxElem;
    std::stringstream cplxStream;

    for (int i = 0; i < shape[0]; i++) {

        std::getline(inFile, elem, '\n');
        lineStream = std::stringstream(elem);

        for (int j = 0; j < shape[1]; j++) {

            std::getline(lineStream, elem, ';');

            cplxStream = std::stringstream(elem);
            std::getline(cplxStream, elem, ',');
            d1 = std::stod(elem);
            std::getline(cplxStream, elem, ',');
            d2 = std::stod(elem);

            out[{i, j}] = std::complex<double>(d1, d2);
        }
    }

    return out;
}

arrcplx paddedfft2NP(arr& in) {

    std::string path_in = "/home/max/projects/apfc/tmp/np_fft/py_in.txt";
    std::string path_out = "/home/max/projects/apfc/tmp/np_fft/py_out.txt";

    write2DToFile(in, path_in);
    std::system("/home/max/projects/apfc/tmp/np_fft/py_fft.sh");
    return readComplex(path_out, in.shape());
}

arr invPaddedfft2NP(arrcplx& in, int sx, int sy) {

    std::string path_in = "/home/max/projects/apfc/tmp/np_fft/py_in.txt";
    std::string path_out = "/home/max/projects/apfc/tmp/np_fft/py_out.txt";

    write2DToFile(in, path_in);
    std::system("/home/max/projects/apfc/tmp/np_fft/py_ifft.sh");
    return readReal(path_out, in.shape());
}

template<class T>
arrcplx getInvFFTForm(T& input) {

    int rows = input.shape()[0];
    int cols = input.shape()[1];

    int isOdd = static_cast<int>(rows % 2 != 0);
    int k = (rows - isOdd) / 2 + 1;

    T zeros = xt::ones<std::complex<double>>({rows, k});

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < k; j++)
            zeros[{i, j}] = input[{i, j}];
    }

    return zeros;
}

arrcplx paddedfft2CS(arr in) {
    //arr pad = padZeros(in);
    arrcplx fft = xt::fftw::rfft2(in);
    return copySwap(fft);
}

arr invPaddedfft2CS(arrcplx& in, int sx, int sy) {
    arrcplx fitArr = getInvFFTForm(in);
    // true so the odd dimension will be returned
    arr ifft = xt::fftw::irfft2(fitArr, true);
    return ifft;
}

arrcplx paddedfft2(arr in) {
    arrcplx fft = xt::fftw::rfft2(in);
    return fft;
}

arr invPaddedfft2(arrcplx& in, int sx, int sy) {
    arrcplx fitArr = arrcplx(xt::view(in, xt::all(), xt::all()));
    arr ifft = xt::fftw::irfft2(fitArr, true);
    return ifft;
}

arr invPaddedfft2(arrcplx& in) {
    arrcplx fitArr = arrcplx(xt::view(in, xt::all(), xt::all()));
    arr ifft = xt::fftw::irfft2(fitArr, true);
    return ifft;
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

        int rows = x.shape()[0];
        int cols = x.shape()[1];
        int isOdd = static_cast<int>(rows % 2 != 0);
        int k = (rows - isOdd) / 2 + 1;

        //kxm = kxm_;
        //kym = kym_;

        kxm = xt::zeros<double>({rows, k});
        kym = xt::zeros<double>({rows, k});

        int kDiff = rows - k;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < k; j++) {
                kxm[{i, j}] = kym_[{i, j}];
                kym[{i, j}] = kxm_[{i, j}];
            }
        }

        //arr kymRow = arr(xt::view(kym, 0, xt::all()));
        //printArr1D(kymRow);
        //printArr(kxm);
        //printArr(kym);
    }

    void initEtas() {

        arr rad = xt::sqrt(xm * xm + ym * ym) - sett->initRadius;
        arr tan_ = 0.5 * (1. + xt::tanh(-3. * rad / sett->interfaceWidth));
        tan_ *= sett->initEta;

        for (int i = 0; i < sett->g.shape()[0]; i++)
            etas.push_back(arr(xt::view(tan_, xt::all(), xt::all())));
    }

    void initGSqHat() {

        arr ksum = kxm * kxm + kym * kym;

        for (int i = 0; i < sett->g.shape()[0]; i++) {

            arr val = arr(ksum);
            val += 2. * sett->g[{i, 0}] * kxm;
            val += 2. * sett->g[{i, 1}] * kym;
            val *= val;
            val = xt::eval(val);

            gSqHat.push_back(arr(xt::view(val, xt::all(), xt::all())));
        }
    }

    arr ampAbsSqSum(int eta_i) {

        int is_eta_i;

        arr sum_ = xt::zeros<double>(etas[0].shape());
        for (int eta_j = 0; eta_j < etas.size(); eta_j++) {

            is_eta_i = (int)(eta_i == eta_j);
            sum_ += (2. - is_eta_i) * etas[eta_j] * etas[eta_j];
        }

        return xt::eval(sum_);
    }

    arrcplx nHat(int eta_i) {

        arr n = xt::ones<double>(etas[0].shape());

        for (int eta_j = 0; eta_j < etas.size(); eta_j++) {
            if (eta_i == eta_j)
                continue;

            n *= arr(xt::view(etas[eta_j], xt::all(), xt::all()));
        }

        n *= 2. * sett->C;
        n += 3. * sett->D * ampAbsSqSum(eta_i) * etas[eta_i];

        arrcplx nn = paddedfft2(n);

        return xt::eval(-1. * sett->gNormSq[eta_i] * nn);
    }

    arrcplx lagrHat(int eta_i) {
        arrcplx lagr = sett->A * gSqHat[eta_i] + sett->B;
        return xt::eval(-1. * lagr * sett->gNormSq[eta_i]);
    }

    arr etaRoutine(int eta_i) {

        arrcplx lagr = lagrHat(eta_i);
        arrcplx n = nHat(eta_i);

        arrcplx exp_lagr = xt::exp(lagr * sett->dt);
        arrcplx fftEta = paddedfft2(etas[eta_i]);

        arrcplx n_eta = exp_lagr * fftEta;
        n_eta += ((exp_lagr - 1.) / lagr) * n;

        arr inEta = invPaddedfft2(n_eta);

        return xt::eval(inEta);
    }

    void runOneStep() {

        std::vector<arr> n_etas;
        for (int eta_i = 0; eta_i < etas.size(); eta_i++)
            n_etas.push_back(arr(etaRoutine(eta_i)));

        etas = std::vector<arr>(n_etas);
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

    std::vector<arr> etas;
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
        /*dt_             =*/ 0.1,
        /*numT_           =*/ 100000,
        /*initRadius_     =*/ 10.,
        /*initEta_        =*/ 0.016,
        /*interfaceWidth_ =*/ 3. * 3.14,
        /*xlim_           =*/ 50.,
        /*numPts_         =*/ 201,
        /*G               =*/ G,
        /*simDir          =*/ "/home/max/projects/apfc/data/sim"
    );

    auto sim = FFTSim(sett);
    sim.run("/home/max/projects/apfc/data/sim", 10);

    auto end = std::chrono::system_clock::now();

    auto timeDiff = end - start;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(timeDiff);
    auto mins = std::chrono::duration_cast<std::chrono::minutes>(timeDiff - hours);
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(timeDiff - hours - mins);

    std::cout << "Time Elapsed: " << hours.count() << ":" << mins.count() << ":" << secs.count() << std::endl;

    return 0;
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
        /*dB0_            =*/ -0.1,//0.04,
        /*dt_             =*/ 0.2,
        /*numT_           =*/ 10000,
        /*initRadius_     =*/ 10.,
        /*initEta_        =*/ 0.01608,
        /*interfaceWidth_ =*/ 3. * 3.14,
        /*xlim_           =*/ 50.,
        /*numPts_         =*/ 201,
        /*G               =*/ G,
        /*simDir          =*/ "/home/max/projects/apfc/data/sim"
    );

    auto sim = FFTSim(sett);

    int eta_i = 0;

    arrcplx lagr = sim.lagrHat(eta_i);
    write2DToFile(lagr, "/home/max/projects/apfc/tmp/comp/cpp_lagr.txt");
    arrcplx n = sim.nHat(eta_i);
    write2DToFile(n, "/home/max/projects/apfc/tmp/comp/cpp_n.txt");

    arrcplx exp_lagr = xt::exp(lagr * sett->dt);
    arrcplx fftEta = paddedfft2(sim.etas[eta_i]);

    write2DToFile(exp_lagr, "/home/max/projects/apfc/tmp/comp/cpp_explagr.txt");

    arrcplx n_eta = exp_lagr * fftEta;
    //write2DToFile(n_eta, "/home/max/projects/apfc/tmp/comp/cpp_neta1.txt");
    n_eta += ((exp_lagr - 1.) / lagr) * n;
    write2DToFile(n_eta, "/home/max/projects/apfc/tmp/comp/cpp_neta2.txt");

    arr inEta = invPaddedfft2(n_eta, sim.xm.shape()[0], sim.xm.shape()[1]);
    write2DToFile(inEta, "/home/max/projects/apfc/tmp/comp/cpp_ifft.txt");

    arr n_ = xt::ones<double>(sim.etas[0].shape());

    for (int eta_j = 0; eta_j < sim.etas.size(); eta_j++) {
        if (eta_i == eta_j)
            continue;

        n_ *= sim.etas[eta_j];
    }

    n_ *= 2. * sett->C;
    write2DToFile(n_, "/home/max/projects/apfc/tmp/comp/cpp_nhatn.txt");

    arr abssum = sim.ampAbsSqSum(eta_i);
    write2DToFile(abssum, "/home/max/projects/apfc/tmp/comp/cpp_nhatabssum.txt");

    arr n2 = 3. * sett->D * abssum * sim.etas[eta_i];
    write2DToFile(n2, "/home/max/projects/apfc/tmp/comp/cpp_nhatn2.txt");

    arr comb = n_ + n2;
    write2DToFile(comb, "/home/max/projects/apfc/tmp/comp/cpp_nhatcomb.txt");

    arrcplx nn = paddedfft2(comb);
    //arrcplx nn = xt::fftw::hfft2(comb);
    //nn = xt::fftw::fftshift(nn);
    write2DToFile(nn, "/home/max/projects/apfc/tmp/comp/cpp_nhatfft.txt");

    arr nhatifft = invPaddedfft2(nn, sim.xm.shape()[0], sim.xm.shape()[1]);
    write2DToFile(nhatifft, "/home/max/projects/apfc/tmp/comp/cpp_nhatifft.txt");

    arrcplx test = xt::fftw::rfft2(sim.etas[0]);
    //test = xt::fftw::fftshift(test);
    //write2DToFile(test, "/home/max/projects/apfc/tmp/comp/cpp_ffttest.txt");

    arrcplx in = {
        {0., 0., 0., 0., 0., 2., 2., 2.},
        {0., 0., 0., 0., 0., 2., 2., 2.},
        {0., 0., 1., 1., 1., 2., 2., 2.},
        {0., 0., 1., 1., 1., 2., 2., 2.},
        {0., 0., 1., 1., 1., 2., 2., 2.}
    };

    in = xt::transpose(in);
    arrcplx out = copySwap(in);

    int rows_in = in.shape()[1];
    int cols_in = in.shape()[0];

    for (int i = 0; i < cols_in; i++) {
        for (int j = 0; j < rows_in; j++) {
            std::cout << xt::real(in[{i, j}]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    //-----------

    int rows_out = out.shape()[1];
    int cols_out = out.shape()[0];

    for (int i = 0; i < cols_out; i++) {
        for (int j = 0; j < rows_out; j++) {
            std::cout << xt::real(out[{i, j}]) << " ";
        }
        std::cout << std::endl;
    }

    in = {
        {0., 0., 0., 0., 2., 2., 2.},
        {0., 1., 1., 1., 2., 2., 2.},
        {0., 1., 1., 1., 2., 2., 2.},
        {0., 1., 1., 1., 2., 2., 2.}
    };

    in = xt::transpose(in);
    out = copySwap(in);

    rows_in = in.shape()[1];
    cols_in = in.shape()[0];

    for (int i = 0; i < cols_in; i++) {
        for (int j = 0; j < rows_in; j++) {
            std::cout << xt::real(in[{i, j}]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    rows_out = out.shape()[1];
    cols_out = out.shape()[0];

    for (int i = 0; i < cols_out; i++) {
        for (int j = 0; j < rows_out; j++) {
            std::cout << xt::real(out[{i, j}]) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

int main3() {

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
        /*dt_             =*/ 0.2,
        /*numT_           =*/ 10000,
        /*initRadius_     =*/ 10.,
        /*initEta_        =*/ 0.01608,
        /*interfaceWidth_ =*/ 3. * 3.14,
        /*xlim_           =*/ 2.,
        /*numPts_         =*/ 5,
        /*G               =*/ G,
        /*simDir          =*/ "/home/max/projects/apfc/data/sim"
    );

    auto sim = FFTSim(sett);

    printArr(sim.kxm);
    printArr(sim.kym);

    auto m = xt::fftw::rfftscale<double>(5);

    std::cout << "1" << std::endl;
    printShape(m);
    std::cout << "2" << std::endl;
    printArr(m);

    return 0;
}
