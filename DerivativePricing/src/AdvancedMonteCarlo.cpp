#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <bits/stdc++.h> 

// Asian Option Pricing w/ Monte Carlo

double norm_cdf(double x) {
    return std::erfc(-x/std::sqrt(2))/2;
}


double GeometricAsianBlackScholes(double S0, double T, double r, 
                                  double K, double sig)
{
    double N = T*252.0;    
    double sig_hat = sig * sqrt( (2*N+1) / (6*(N+1)) );
    double rho = 0.5*(r - 0.5*sig*sig + sig_hat*sig_hat);
    double d1 = (1/(sqrt(T)*sig_hat)) * (log(S0/K) + (rho + 0.5*sig_hat*sig_hat)*T);
    double d2 = (1/(sqrt(T)*sig_hat)) * (log(S0/K) + (rho - 0.5*sig_hat*sig_hat)*T);
    double CallPrice = exp(-r*T) * (S0*exp(rho*T)*norm_cdf(d1) - K*norm_cdf(d2));
    return CallPrice;
}

struct MonteCarlo {
    double val;    
    double SD;
    double SE; 
    double interval_lower;
    double interval_upper;
};

MonteCarlo ArithmeticAsianCall(double S0, double T, double r, 
                               double K, double sig, int M)
{
    double N = T*252.0;
    double dt = T/N;    
    double nudt = (r-0.5*sig*sig)*dt;
    double sigsdt = sig*sqrt(dt);

    double sum_CT = 0.0;
    double sum_CT2 = 0.0;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(0.0,1.0);

    for (int i = 0; i < M; i++) {

        double St = S0;
        double sumSt = S0;

        for (int j = 0; j < N; j++) { 
            double epsilon = distribution(generator);
            St = St * exp(nudt + sigsdt*epsilon);
            sumSt = sumSt + St;
        }
        
        double CT = 0.0;
        double A = sumSt/(N+1);
        CT = (A - K > 0.0 ? A - K : 0.0);
        sum_CT += CT;
        sum_CT2 += CT*CT;
    }

    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    // 95 percent confidence interval
    MC.interval_lower = MC.val - 1.96*MC.SD/sqrt(M);
    MC.interval_upper = MC.val + 1.96*MC.SD/sqrt(M);
    return MC;
}

MonteCarlo GeometricAsianCall(double S0, double T, double r, 
                               double K, double sig, int M)
{
    double N = T*252.0;
    double dt = T/N;    
    double nudt = (r-0.5*sig*sig)*dt;
    double sigsdt = sig*sqrt(dt);

    double sum_CT = 0.0;
    double sum_CT2 = 0.0;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(0.0,1.0);

    for (int i = 0; i < M; i++) {

        double St = S0;
        double powprodSt = pow(S0, 1.0/(N+1.0));

        for (int j = 0; j < N; j++) { 
            double epsilon = distribution(generator);
            St = St * exp(nudt + sigsdt*epsilon);
            powprodSt = powprodSt * pow(St, 1.0/(N+1.0));
        }
        
        double CT = 0.0;
        double G = powprodSt;
        CT = (G - K > 0.0 ? G - K : 0.0);
        sum_CT += CT;
        sum_CT2 += CT*CT;
    }

    MonteCarlo MC;
    MC.val = (sum_CT / (double)M) * exp(-r*T);
    MC.SD = sqrt((sum_CT2 - sum_CT*sum_CT/(double)M) * exp(-2.0*r*T) / ((double)M-1.0));
    MC.SE = MC.SD / sqrt((double)M);
    // 95 percent confidence interval
    MC.interval_lower = MC.val - 1.96*MC.SD/sqrt(M);
    MC.interval_upper = MC.val + 1.96*MC.SD/sqrt(M);
    return MC;
}

struct LinearReg {
    double X;   
    double Y;
    double b_star; 
};


LinearReg RegAsianCall(double S0, double T, double r, 
                       double K, double sig, int M)
{
    double N = T*252.0;
    double dt = T/N;    
    double nudt = (r-0.5*sig*sig)*dt;
    double sigsdt = sig*sqrt(dt);

    double sum_CTA = 0.0;
    double sum_CTG = 0.0;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(0.0,1.0);

    double CTA[M];
    double CTG[M];

    for (int i = 0; i < M; i++) {

        double St = S0;
        double sumSt = S0;
        double powprodSt = pow(S0, 1.0/(N+1.0));

        for (int j = 0; j < N; j++) { 
            double epsilon = distribution(generator);
            St = St * exp(nudt + sigsdt*epsilon);
            sumSt = sumSt + St;
            powprodSt = powprodSt * pow(St, 1.0/(N+1.0));
        }

        double A = sumSt/(N+1);
        CTA[i] = (A - K > 0.0 ? A - K : 0.0);
        double G = powprodSt;
        CTG[i] = (G - K > 0.0 ? G - K : 0.0);
        sum_CTA += CTA[i];
        sum_CTG += CTG[i];
    }

    double X_bar = sum_CTA / (double)M;
    double Y_bar = sum_CTG / (double)M;

    double nume = 0.0;
    double deno = 0.0;
    for (int i = 0; i < M; i++) {
        nume += (CTA[i]-X_bar)*(CTG[i]-Y_bar);
        deno += (CTA[i]-X_bar)*(CTA[i]-X_bar);
    }

    LinearReg LR;
    LR.X = (sum_CTA / (double)M) * exp(-r*T);
    LR.Y = (sum_CTG / (double)M) * exp(-r*T);
    LR.b_star = nume/deno;
    
    return LR;
}

int main(int argc, char *argv[]) {

    double Pg = GeometricAsianBlackScholes(100.0, 5.0, 0.25/100.0, 100.0, 0.4);
    std::cout << "Black-Scholes Geometric Asian Call: " << Pg << std::endl;
    std::cout << "\n" << std::endl;

    MonteCarlo result[2];
    clock_t start[2];
    clock_t stop[2];
    double time[2];
    const char* meanType[3] = {"Arithmetic", "Geometric"};

    start[0] = clock();
    result[0] = ArithmeticAsianCall(100.0, 5.0, 0.25/100.0, 100.0, 0.4, 1000000);
    stop[0] = clock();
    time[0] = (stop[0] - start[0]) / double(CLOCKS_PER_SEC);

    start[1] = clock();
    result[1] = GeometricAsianCall(100.0, 5.0, 0.25/100.0, 100.0, 0.4, 1000000);
    stop[1] = clock();
    time[1] = (stop[1] - start[1]) / double(CLOCKS_PER_SEC);

    printf("%-10s %-10s %-24s %-10s %-10s\n", 
            "mean type", "price", "conf. interval (95%)", "st. err", "runtime (secs)");
    for (int i = 0; i < 2; i++) {
        printf("%-10s %-10f [%-9f, %-9f]   %-10f %-10f\n",
               meanType[i], result[i].val, result[i].interval_lower,
               result[i].interval_upper, result[i].SE, time[i]);
    }
    printf("\n");
    
    LinearReg op;
    op = RegAsianCall(100.0, 5.0, 0.25/100.0, 100.0, 0.4, 10000);
    std::cout << "(M = 10000) b*: " << op.b_star << std::endl;
    std::cout << "(M = 10000) P_arithmetic: " << op.X << std::endl;
    std::cout << "(M = 10000) P_geometric: " << op.Y << std::endl;
    std::cout << "\n" << std::endl;

    double Eg1 = Pg - result[1].val;
    std::cout << "Error of Pricing Geometric Asian: " << Eg1 << std::endl;    

    LinearReg output[9];
    int M[9] = {5000, 8000, 10000, 20000, 50000, 80000, 100000, 200000, 500000};
    double Pa_star[9];
    double Eg[9];
    double diff[9];
    for (int i = 0; i < 9; i++) {
        output[i] = RegAsianCall(100.0, 5.0, 0.25/100.0, 100.0, 0.4, M[i]);
        Eg[i] = Pg - output[i].Y;
        Pa_star[i] = output[i].X - output[i].b_star*Eg[i];
        diff[i] = fabs(result[0].val - Pa_star[i]);
    }

    printf("%-10s %-10s %-10s\n", 
           "M", "price", "difference");
    for (int i = 0; i < 9; i++) {
        printf("%-10i %-10f %-10f\n",
               M[i], Pa_star[i], diff[i]);
    }
    printf("\n");

    return 0;
}
