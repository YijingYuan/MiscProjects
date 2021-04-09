#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <thread>
#include <bits/stdc++.h> 

#ifdef USE_OPENMP
    #include <omp.h>
#endif

// Monte Carlo and Variance Reduction Methods

double norm_cdf(double x) {
    return std::erfc(-x/std::sqrt(2))/2;
}

double BlackScholesDelta(double S, double t, double r, double K, 
                    double sig, double div, char o)
{    
    double d1 = (log(S/K) + (r-div-0.5*sig*sig)*t) / (sig*sqrt(t));
    double d2 = d1 - sig*sqrt(t);
    double result;
    if (o == 'c'){
        result = exp(-div*t)*norm_cdf(d1);
    }
    if (o == 'p'){
        result = exp(-div*t)*(norm_cdf(d1)-1);
    }
    return result;
}

struct MonteCarlo {
    double val;    
    double SD;
    double SE; 
};

// Basic Monte Carlo Scheme

MonteCarlo Vanilla(char opt, int N, int M, bool print)
{
    double sum_CT = 0.0;
    double sum_CT2 = 0.0;
    double r = 0.06;    
    double T = 1.0;


#ifdef USE_OPENMP
    #pragma omp parallel firstprivate(opt, N, M, T, r) shared(sum_CT, sum_CT2)
#endif
    {
        double K = 100;
        double S0 = 100;
        double sig = 0.2;
        double div = 0.03;

        double dt = T/(double)N;
        double nudt = (r-div-0.5*sig*sig)*dt;
        double sigsdt = sig*sqrt(dt);
        double lnS = log(S0);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        seed += std::hash<std::thread::id>()(std::this_thread::get_id());
        std::default_random_engine generator (seed);
        std::normal_distribution<double> distribution(0.0,1.0);
#ifdef USE_OPENMP
        #pragma omp for
#endif
        for (int i = 0; i < M; i++) {
            if (print && (i % 20000) == 0 && i != 0)
                std::cout << "i: " <<  i << std::endl;

            double lnSt = lnS;
            for (int j = 0; j < N; j++) {
                double epsilon = distribution(generator);
                lnSt = lnSt + nudt + sigsdt*epsilon;
            }
            double ST = exp(lnSt);
            double CT = 0.0;
            if (opt == 'c') {
                CT = ST - K;
                if (CT < 0) {
                    CT = 0;
                }
            }
            if (opt == 'p') {
                CT = K - ST;
                if (CT < 0) {
                    CT = 0;
                }
            }

#ifdef USE_OPENMP
            #pragma omp critical
#endif
            {
                sum_CT += CT;
                sum_CT2 += CT*CT;
            }
        }
    }

    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    return MC;
}

// Monte Carlo with Antithetic Variance

MonteCarlo Antithetic(char opt, int N, int M, bool print)
{
    double sum_CT = 0.0;
    double sum_CT2 = 0.0;
    double r = 0.06;
    double T = 1.0;

#ifdef USE_OPENMP
    #pragma omp parallel firstprivate(opt, N, M, T, r) shared(sum_CT, sum_CT2)
#endif
    {
        double K = 100;
        double S0 = 100;
        double sig = 0.2;
        double div = 0.03;

        double dt = T/(double)N;
        double nudt = (r-div-0.5*sig*sig)*dt;
        double sigsdt = sig*sqrt(dt);
        double lnS = log(S0);

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        seed += std::hash<std::thread::id>()(std::this_thread::get_id());
        std::default_random_engine generator (seed);
        std::normal_distribution<double> distribution(0.0,1.0);
#ifdef USE_OPENMP
        #pragma omp for
#endif
        for (int i = 0; i < M; i++) {
            if (print && (i % 50000) == 0 && i != 0)
                std::cout << "i: " <<  i << std::endl;

            double lnSt1 = lnS;
            double lnSt2 = lnS;
            for (int j = 0; j < N; j++) {
                double epsilon = distribution(generator);
                lnSt1 = lnSt1 + nudt + sigsdt*epsilon;
                lnSt2 = lnSt2 + nudt + sigsdt*(-epsilon);
            }
            double ST1 = exp(lnSt1);
            double ST2 = exp(lnSt2);
            double CT = 0.0;
            double CT1 = 0.0;
            double CT2 = 0.0;
            if (opt == 'c') {
                CT1 = (ST1 - K > 0.0 ? ST1 - K : 0.0);
                CT2 = (ST2 - K > 0.0 ? ST2 - K : 0.0);
                CT = 0.5 * (CT1 + CT2);
            }

            if (opt == 'p') {
                CT1 = (K - ST1 > 0.0 ? K - ST1 : 0.0);
                CT2 = (K - ST2 > 0.0 ? K - ST2 : 0.0);
                CT = 0.5 * (CT1 + CT2);
            }

#ifdef USE_OPENMP
            #pragma omp critical
#endif
            {
                sum_CT += CT;
                sum_CT2 += CT*CT;
            }
        }
    }

    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    return MC;
}

// Monte Carlo with Delta-based Control Variate

MonteCarlo DeltaControl(char opt, int N, int M, bool print)
{
        
    double sum_CT = 0.0;
    double sum_CT2 = 0.0;
    double r = 0.06;
    double T = 1.0;

#ifdef USE_OPENMP
    #pragma omp parallel firstprivate(opt, N, M, T, r) shared(sum_CT, sum_CT2)
#endif
    {
        double K = 100;
        double S0 = 100;
        double sig = 0.2;
        double div = 0.03;

        double dt = T/(double)N;
        double nudt = (r-div-0.5*sig*sig)*dt;
        double sigsdt = sig*sqrt(dt);
        double erddt = exp((r-div)*dt);

        double beta1 = -1.0;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        seed += std::hash<std::thread::id>()(std::this_thread::get_id());
        std::default_random_engine generator (seed);
        std::normal_distribution<double> distribution(0.0,1.0);
#ifdef USE_OPENMP
        #pragma omp for
#endif
        for (int i = 0; i < M; i++) {
            double St = S0;
            double CV = 0.0;
            double CT = 0.0;
            for (int j = 0; j < N; j++) {
                double t = (j-1)*dt;
                double delta = BlackScholesDelta(St, T, r, K, sig, div, opt);
                double epsilon = distribution(generator);
                double Stn = St * exp(nudt + sigsdt*epsilon);
                CV = CV + delta*(Stn-St*erddt);
                St = Stn;
            }

            if (opt == 'c') {
                CT = (St - K > 0.0 ? St - K : 0.0) + beta1*CV;
            }

            if (opt == 'p') {
                CT = (K - St > 0.0 ? K - St : 0.0) + beta1*CV;
            }

#ifdef USE_OPENMP
            #pragma omp critical
#endif
            {
                sum_CT += CT;
                sum_CT2 += CT*CT;
            }
        }
    }

    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    return MC;
}

// Monte Carlo with Antithetic and Delta-based Control Variates

MonteCarlo AntitheticDeltaControl(char opt, int N, int M, bool print)
{
    double sum_CT = 0.0;
    double sum_CT2 = 0.0;
    double r = 0.06;
    double T = 1.0;

#ifdef USE_OPENMP
    #pragma omp parallel firstprivate(opt, N, M, T, r) shared(sum_CT, sum_CT2)
#endif
    {
        double K = 100.0;
        double S0 = 100.0;
        double sig = 0.2;
        double div = 0.03;

        double dt = T/(double)N;
        double nudt = (r-div-0.5*sig*sig)*dt;
        double sigsdt = sig*sqrt(dt);
        double erddt = exp((r-div)*dt);

        double beta1 = -1.0;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        seed += std::hash<std::thread::id>()(std::this_thread::get_id());
        std::default_random_engine generator (seed);
        std::normal_distribution<double> distribution(0.0,1.0);
#ifdef USE_OPENMP
        #pragma omp for
#endif
        for (int i = 0; i < M; i++) {
            if (print && (i % 50000) == 0 && i != 0)
                std::cout << "i: " <<  i << std::endl;

            double St1 = S0;
            double St2 = S0;
            double CV1 = 0.0;
            double CV2 = 0.0;
            for (int j = 0; j < N; j++) {
                double t = (j-1)*dt;
                double delta1 = BlackScholesDelta(St1, T, r, K, sig, div, opt);
                double delta2 = BlackScholesDelta(St2, T, r, K, sig, div, opt);
                double epsilon = distribution(generator);
                double Stn1 = St1 * exp(nudt + sigsdt*epsilon);
                double Stn2 = St2 * exp(nudt + sigsdt*(-epsilon));
                CV1 = CV1 + delta1 * (Stn1-St1*erddt);
                CV2 = CV2 + delta2 * (Stn2-St2*erddt);
                St1 = Stn1;
                St2 = Stn2;
            }
            
            double CT = 0.0;
            double CT1 = 0.0;
            double CT2 = 0.0;
            if (opt == 'c') {
                CT1 = (St1 - K > 0.0 ? St1 - K : 0.0) + beta1*CV1;
                CT2 = (St2 - K > 0.0 ? St2 - K : 0.0) + beta1*CV2;
                CT = 0.5 * (CT1 + CT2);
            }

            if (opt == 'p') {
                CT1 = (K - St1 > 0.0 ? K - St1 : 0.0) + beta1*CV1;
                CT2 = (K - St2 > 0.0 ? K - St2 : 0.0) + beta1*CV2;
                CT = 0.5 * (CT1 + CT2);
            }

#ifdef USE_OPENMP
            #pragma omp critical
#endif
            {
                sum_CT += CT;
                sum_CT2 += CT*CT;
            }
        }
    }

    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    return MC;
}


// Pricing Asian Barrier Option Using Monte Carlo

MonteCarlo AsianUpOut(char opt, int day, int M, bool print)
{
    double sum_CT = 0.0;
    double sum_CT2 = 0.0;
    double K = 100.0;
    double S0 = 100.0;
    double sig = 0.2;
    double div = 0.03;
    double r = 0.06;
    double T = 1.0/6.0;

    int N = day*24;
    double dt = T/(double)N;    
    double nudt = (r-div-0.5*sig*sig)*dt;
    double sigsdt = sig*sqrt(dt);

#ifdef USE_OPENMP
    #pragma omp parallel firstprivate(opt, day, M, print, K, S0, sig, div, r, T, N, dt, nudt, sigsdt) shared(sum_CT, sum_CT2)
#endif
    {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        seed += std::hash<std::thread::id>()(std::this_thread::get_id());
        std::default_random_engine generator (seed);
        std::normal_distribution<double> distribution(0.0,1.0);

#ifdef USE_OPENMP
        #pragma omp for
#endif
        for (int i = 1; i <= M; i++) {
            if (print && (i % 50000) == 0 && i != 0)
                std::cout << "i: " <<  i << std::endl;
            double St = S0;
            double sumSt = 0.0;
            bool IN = true;

            for (int j = 1; j <= N; j++) { 
                double epsilon = distribution(generator);
                St = St * exp(nudt + sigsdt*epsilon);
                if (j % 24 == 0) {
                    sumSt = sumSt + St;
                }
                if (St >= 110) {
                    IN = false;
                    break;
                }
            }
            
            double CT = 0.0;
            double A = sumSt/day;
            if (IN) {
                if (opt == 'c') {
                    CT = (A - K > 0.0 ? A - K : 0.0);
                }
                if (opt == 'p') {
                    CT = (K - A > 0.0 ? K - A : 0.0);
                }
            }
#ifdef USE_OPENMP
            #pragma omp atomic
#endif
            sum_CT += CT;
#ifdef USE_OPENMP
            #pragma omp atomic
#endif
            sum_CT2 += CT*CT;
        }
    }
    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    return MC;
}



// Pricing Futures Portfolio Using Euler/Milstein Methods

MonteCarlo FutureSpread(double T, int N, int M) {

    double r = 0.0;
    double dt = T/(double)N;
    double sum_CT = 0.0;
    double sum_CT2 = 0.0;

#ifdef USE_OPENMP
    #pragma omp parallel firstprivate(M, r, T, N, dt) shared(sum_CT, sum_CT2)
#endif
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed += std::hash<std::thread::id>()(std::this_thread::get_id());
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(0.0,1.0);

#ifdef USE_OPENMP
        #pragma omp for
#endif
    for (int i = 0; i < M; i++) {

        double xt1 = 1000.0;
        double vt = 0.16;
        double xt2 = 1000.0; 

        for (int j = 0; j < N; j++) {
            double epsilon1 = distribution(generator);
            xt1 = xt1 + 0.001*xt1*dt + 0.1*xt1*sqrt(dt)*epsilon1;
            double epsilon2 = distribution(generator);
            vt = vt + 10.0*(0.16-vt)*dt + 0.3*sqrt(vt)*sqrt(dt)*epsilon2;
            double epsilon3 = distribution(generator);
            xt2 = xt2 + 0.01*xt2*dt + sqrt(vt)*xt2*sqrt(dt)*epsilon3;
        }
        double CT = xt1-xt2;

#ifdef USE_OPENMP
        #pragma omp critical
#endif
        {
            sum_CT = sum_CT + CT;
            sum_CT2 = sum_CT2 + CT*CT;
        }
    }
}
    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    return MC;
}

// Problem 3. (Bonus) - Correlated Brownian Motions

MonteCarlo CorrFutures_Euler(double T, int N, int M) {

    double r = 0.0;
    double dt = T/(double)N;
    double a = 0.4; 
    double b = -0.6;
    double sum_CT = 0.0;
    double sum_CT2 = 0.0;

#ifdef USE_OPENMP
    #pragma omp parallel firstprivate(M, r, T, N, dt) shared(sum_CT, sum_CT2)
#endif
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed += std::hash<std::thread::id>()(std::this_thread::get_id());
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(0.0,1.0);

#ifdef USE_OPENMP
        #pragma omp for
#endif
    for (int i = 0; i < M; i++) {
        double xt1 = 1000.0;
        double vt = 0.16;
        double xt2 = 1000.0; 
        for (int j = 0; j < N; j++) {
            double epsilon2 = distribution(generator);
            double epsilon3 = distribution(generator);
            double epsilon4 = distribution(generator);
            double dwt2 = sqrt(dt)*epsilon2;
            double dwt3 = sqrt(dt)*epsilon3;
            double dwt4 = sqrt(dt)*epsilon4;
            double dwt1 = a*dwt2 + sqrt(1-a*a)*dwt3;
            double dzt = b*dwt2 + sqrt(1-b*b)*dwt4;
            xt1 = xt1 + 0.001*xt1*dt + 0.1*xt1*dwt1;
            vt = vt + 10.0*(0.16-vt)*dt + 0.3*sqrt(vt)*dzt;
            xt2 = xt2 + 0.01*xt2*dt + sqrt(vt)*xt2*dwt2;
        }
        double CT = xt1-xt2;

#ifdef USE_OPENMP
        #pragma omp critical
#endif
        {
            sum_CT = sum_CT + CT;
            sum_CT2 = sum_CT2 + CT*CT;
        }
    }
}
    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    return MC;
}


MonteCarlo CorrFutures_Milstein(double T, int N, int M) {

    double r = 0.0;
    double dt = T/(double)N;
    double a = 0.4; 
    double b = -0.6;
    double sum_CT = 0.0;
    double sum_CT2 = 0.0;

#ifdef USE_OPENMP
    #pragma omp parallel firstprivate(M, r, T, N, dt) shared(sum_CT, sum_CT2)
#endif
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    seed += std::hash<std::thread::id>()(std::this_thread::get_id());
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(0.0,1.0);

#ifdef USE_OPENMP
        #pragma omp for
#endif
    for (int i = 0; i < M; i++) {
        double xt1 = 1000.0;
        double vt = 0.16;
        double xt2 = 1000.0; 
        for (int j = 0; j < N; j++) {
            double epsilon2 = distribution(generator);
            double epsilon3 = distribution(generator);
            double epsilon4 = distribution(generator);
            double dwt2 = sqrt(dt)*epsilon2;
            double dwt3 = sqrt(dt)*epsilon3;
            double dwt4 = sqrt(dt)*epsilon4;
            double dwt1 = a*dwt2 + sqrt(1-a*a)*dwt3;
            double dzt = b*dwt2 + sqrt(1-b*b)*dwt4;
            xt1 = xt1 + 0.001*xt1*dt + 0.1*xt1*dwt1
                      + 0.5*0.1*0.1*xt1*(dwt1*dwt1-dt);
            vt = vt + 10.0*(0.16-vt)*dt + 0.3*sqrt(vt)*dzt
                    + 0.25*0.3*0.3*(dzt*dzt-dt);
            xt2 = xt2 + 0.01*xt2*dt + sqrt(vt)*xt2*dwt2
                      + 0.5*vt*xt2*(dwt2*dwt2-dt);
        }
        double CT = xt1-xt2;

#ifdef USE_OPENMP
        #pragma omp critical
#endif
        {
            sum_CT = sum_CT + CT;
            sum_CT2 = sum_CT2 + CT*CT;
        }
    }
}
    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    return MC;
}


int main(int argc, char *argv[]) {
    bool print = (argc > 1);
#ifdef USE_OPENMP
    if (print)
        std::cout << "Using openmp\n";
#endif
    // Monte Carlo Schemes Output 
    MonteCarlo result[10];
    clock_t start[10];
    clock_t stop[10];
    double time[10];

    int N[4] = {300, 600, 300, 600};
    int M[4] = {10, 100, 200, 200};

    start[0] = clock();
    result[0] = Vanilla('c', N[0], M[0], print);
    stop[0] = clock();
    time[0] = (stop[0] - start[0]) / double(CLOCKS_PER_SEC);

    start[1] = clock();
    result[1] = Vanilla('c', N[1], M[1], print);
    stop[1] = clock();
    time[1] = (stop[1] - start[1]) / double(CLOCKS_PER_SEC);

    start[2] = clock();
    result[2] = Vanilla('c', N[2], M[2], print);
    stop[2] = clock();
    time[2] = (stop[2] - start[2]) / double(CLOCKS_PER_SEC);

    start[3] = clock();
    result[3] = Vanilla('c', N[3], M[3], print);
    stop[3] = clock();
    time[3] = (stop[3] - start[3]) / double(CLOCKS_PER_SEC);

    printf("%-7s %-10s %-10s %-10s %-10s %-10s\n", 
            "N", "M", "option", "st. dev.", "st. error", "runtime (secs)");
    for (int i = 0; i < 4; i++) {
        printf("%-7i %-10i %-10f %-10f %-10f %-10f\n",
            N[i], M[i], result[i].val, result[i].SD, result[i].SE, time[i]);
    }
    printf("\n");

    MonteCarlo test;
    test = Vanilla('c', 10, 100, print);
    std::cout << test.val << " " << test.SD << " " << test.SE << std::endl;

    const char* CV[3] = {"Antithetic", "Delta Control", "Both"};
    
    start[4] = clock();
    result[4] = Antithetic('c', 600, 1000, print);
    stop[4] = clock();
    time[4] = (stop[4] - start[4]) / double(CLOCKS_PER_SEC);

    start[5] = clock();
    result[5] = DeltaControl('c', 600, 1000, print);
    stop[5] = clock();
    time[5] = (stop[5] - start[5]) / double(CLOCKS_PER_SEC);

    start[6] = clock();
    result[6] = AntitheticDeltaControl('c', 600, 1000, print);
    stop[6] = clock();
    time[6] = (stop[6] - start[6]) / double(CLOCKS_PER_SEC);
    
    printf("Call Option:\n");
    printf("%-20s %-10s %-10s %-10s %-10s\n", 
            "Control Variates", "option", "st. dev.", "st. error", "runtime (secs)");
    for (int i = 4; i < 7; i++) {
        printf("%-20s %-10f %-10f %-10f %-10f\n",
            CV[i-4], result[i].val, result[i].SD, result[i].SE, time[i]);
    }
    printf("\n");

    start[7] = clock();
    result[7] = Antithetic('p', 600, 1000, print);
    stop[7] = clock();
    time[7] = (stop[7] - start[7]) / double(CLOCKS_PER_SEC);

    start[8] = clock();
    result[8] = DeltaControl('p', 600, 100, print);
    stop[8] = clock();
    time[8] = (stop[8] - start[8]) / double(CLOCKS_PER_SEC);

    start[9] = clock();
    result[9] = AntitheticDeltaControl('p', 600, 100, print);
    stop[9] = clock();
    time[9] = (stop[9] - start[9]) / double(CLOCKS_PER_SEC);
    
    printf("Put Option:\n");
    printf("%-20s %-10s %-10s %-10s %-10s\n", 
            "Control Variates", "option", "st. dev.", "st. error", "runtime (secs)");
    for (int i = 7; i < 10; i++) {
        printf("%-20s %-10f %-10f %-10f %-10f\n",
            CV[i-7], result[i].val, result[i].SD, result[i].SE, time[i]);
    }
    printf("\n");

    // Asian Barrier Option Output
    MonteCarlo result_exotic;
    result_exotic = AsianUpOut('c', 60, 6000000, print);   //60 days
    std::cout << "Asian Barrier Up and Out: " << result_exotic.val 
              << " st. dev.: " << result_exotic.SD  
              << " st. error: " << result_exotic.SE << std::endl;
    printf("\n");

    // Future Spread Output
    MonteCarlo result1;
    MonteCarlo result2;
    MonteCarlo result3;
    result1 = FutureSpread(2.0/12.0, 600, 3000);
    result2 = CorrFutures_Euler(2.0/12.0, 600, 300);
    result3 = CorrFutures_Milstein(2.0/12.0, 600, 3000);

    std::cout << "Futures Spread (Euler): " << result1.val 
              << " st. dev.: " << result1.SD  
              << " st. error: " << result1.SE << std::endl;
    std::cout << "Correlated Futures Spread (Euler): " << result2.val 
                 << " st. dev.: " << result2.SD 
                 << " st. error: " << result2.SE << std::endl;
    std::cout << "Correlated Futures Spread (Euler-Milstein): " 
                 << result3.val << " st. dev.: " << result3.SD 
                 << " st. error: " << result3.SE << std::endl;;
    return 0;
}

