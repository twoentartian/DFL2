

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <assert.h>
#include <iterator>
#include <string.h>
//#include <hash_set>
//#include <hash_map>
#include <time.h>


#define max_nodes 5000001
#define max_edges 16000000
#define max_degree 100000
#define max_cent 100000000
#define max_distance 1e48
#define max_iter 2000
#define max_shells 200
#define PI 3.14159265


using namespace std;
//using namespace __gnu_cxx;

float ran1(long *idum);
inline double distH2(double r1, double angle1, double r2, double angle2, double zeta);
inline double radial_generator(double k0, double kmax, double gamma, long * idum);
inline double conn_prob(double aux, double mu, double T);
void print(vector<double> vv1, vector<double> vv2, char filename[]);

int main(int argc, char *argv[])
{
    int i, j, k;
    long idum = -12;
    double Integral, c;
    double aux, delta;
    double avg;
    double gamma = 2.1, alpha;
    double R0 = 1, Rh;
    double s0 = 1;
    int N = 1000;
    double avg1 = 11.;
    double T = 0.5;
    double mu, dist;
    double prob;
    vector<double> radial;
    vector<double> angle;
    
    
    if (argc > 0)
    {
        if (argc >= 7)
        {
            N = atoi(argv[1]);
            //			cout << "N = " << N << endl;
            R0 = atof(argv[2]);
            //			cout << "k0 = " << k0 << endl;
            gamma = atof(argv[3]);
            //			cout << "gamma1 = " << gamma1 << endl;
            avg = atof(argv[4]);
            //			cout << "avg1 = " << avg1 << endl;
            T = atof(argv[5]);
            //			cout << "beta = " << beta << endl;
            idum = atoi(argv[6]);
            //			cout << "idum = " << idum << endl;
        }
        else
        {
            cout << "Program generates H2 model." << endl;
            cout << "Program takes 8 parameters." << endl;
            cout << "Entered " << argc << " parameters: "<< endl;
            for (i = 0; i < argc; i++)
            {
                cout << argv[i] << endl;
            }
            cout << "1) number of nodes" << endl;
            cout << "2) smallest radius" << endl;
            cout << "3) gamma" << endl;
            cout << "4) avg (target average degree)" << endl;
            cout << "5) T temperature" << endl;
            cout << "6) random seed" << endl;
            cout << "7) variable output file (optional) " << endl;
            return -1;
        }
    }
    //*************************************
    // calculate parameters
    if (T < 1)
    {
        Integral = PI * T/ sin(PI * T);
        alpha =(gamma - 1 )/2;
        if (gamma > 2)
        {
            c = (avg * PI *(alpha - 0.5) * (alpha - 0.5)) / (2* Integral * alpha * alpha);
            assert(c > 0);
            Rh = 2 * log(N/c);
            mu = Rh;
        }
        else
        {
            c = (avg * PI *(alpha - 0.5) * (alpha - 0.5)) / (2* Integral * alpha * alpha) * exp((2* alpha - 1) * log(N))* exp(R0*(0.5-alpha));
            c = exp ((1./(2*alpha))*log(c));
            assert(c > 0);
            Rh = 2 * log(N/c);
            mu = 2*alpha*(Rh-R0)+R0;

//		cout << "I = " << Integral << endl;
//		cout << "alpha = " << alpha << endl;
//		cout << "avgk = " << avg << endl;
//		cout << "c = " << c << endl;
//		cout << "Rh = " << Rh << endl;
//		cout << "mu = " << mu << endl;
//		return 0;



//        return 0;
        }
        assert(alpha > 0);
        //*************************************
        // assign coordinates
        assert(N > 1);
        radial.resize(N);
        angle.resize(N);
        for (i = 0; i < N; i++)
        {
            radial[i] = radial_generator(R0, Rh, alpha, &idum);
            angle[i] = 2 * PI * ran1(&idum);
        }
        if (argc ==8)
        {
            print(radial, angle, argv[7]);
        }
        //*************************************
        // connection probabilities


//    cout << "N = " << N << endl;
//    cout << "alpha = " << alpha << endl;
//    cout << "I = " << Integral << endl;
//    cout << "Rh = " << Rh << endl;
//    cout << "c = " << c << endl;
//    cout << "mu = " << mu << endl;
//    cout << "avg =  " << avg << endl;
        
        
        for (i = 0; i < N; i++)
        {
            for (j = i + 1; j  < N; j++)
            {
                aux = distH2(radial[i], angle[i], radial[j], angle[j], 1.);
//            conn_prob(double dist, double mu, double T)
                prob = conn_prob(aux, mu, T);
                if (prob > ran1(&idum))
                {
                    cout << i <<  "\t" << j << endl;
                }
            }
        }
    }
    else // T > 1 hot regime
    {
        assert(T > 0);
        assert(gamma > 2);
        alpha = (gamma - 1) / T;
        //cout << "alpha = " << alpha << endl;
        assert(alpha > 1);
        //cout << "c = " << c << endl;
        //cout << "avg = " << avg << endl;
        c = exp((1./T)*log(2.)) * exp((1 - (1./T))*log(PI)) * alpha * alpha * T * T * T / ( 2 * PI* (T - 1) * (alpha * T - 1)* (alpha * T - 1));
        //cout << "c = " << c << endl;
        c = avg / c;
        //cout << "c = " << c << endl;
        Rh = T * log(N /c);
        mu = Rh;
        R0 = 0;
        
        
        
        //cout << "done with coordinate assignment" << endl;;
        radial.resize(N);
        angle.resize(N);
        for (i = 0; i < N; i++)
        {
            radial[i] = 2 * radial_generator(R0, Rh, alpha, &idum);
            angle[i] = 2 * PI * ran1(&idum);
        }
        Rh = 2* Rh;
        mu = Rh;
        //cout << "Rh = " << Rh << endl;
        //cout << "done with coordinate assignment" << endl;;
        if (argc == 8)
        {
            print(radial, angle, argv[7]);
        }
        //cout << "done with coordinate assignment" << endl;;
        for (i = 0; i < N; i++)
        {
            for (j = i + 1; j  < N; j++)
            {
                aux = distH2(radial[i], angle[i], radial[j], angle[j], 1.);
                prob = conn_prob(aux, mu, T);
                if (prob > ran1(&idum))
                {
                    cout << i <<  "\t" << j << endl;
                }
            }
        }
        
        
        return 0;
    }
    
    
    return 0;
}

inline double distH2(double r1, double angle1, double r2, double angle2, double zeta)
{
    double angle = fabs(PI- fabs(PI - fabs(angle1 - angle2)));
    if (angle < 0.00000001)
    {
        //		cout << "1" << endl;
        return fabs(r1 - r2);
    }
    double dist;
    dist = (cosh(zeta*r1)*cosh(zeta*r2)) - (sinh(zeta*r1)*sinh(zeta*r2) * cos(angle));
    dist = acosh(dist);
    dist /= zeta;
    return dist;
    
}



inline double radial_generator(double R0, double Rh, double alpha, long * idum)
{
    // amples radial components from exponential distribution
    double x = ran1(idum);
    assert(alpha > 0);
    return (1/alpha)* log(   exp(alpha*R0)  + x * (exp(alpha*Rh) - exp(alpha*R0)));
}

inline double conn_prob(double dist, double mu, double T)
{
    return 1./(1.+ exp( (dist - mu) / (2*T) ));
    
}

//**********************************************************************

#define IA 16807

#define IM 2147483647

#define AM (1.0/IM)

#define IQ 127773

#define IR 2836

#define NTAB 32

#define NDIV (1+(IM-1)/NTAB)

#define EPS 1.2e-7

#define RNMX (1.0-EPS)

float ran1(long *idum)


/*Minimal" random number generator of Park and Miller with Bays-Durham shuffle and added
 
 safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive of the endpoint
 
 values). Call with idum a negative integer to initialize; thereafter, do not alter idum between
 
 successive deviates in a sequence. RNMX should approximate the largest floating value that is
 
 less than 1.*/

{
    
    int j;
    
    long k;
    
    static long iy=0;
    
    static long iv[NTAB];
    
    float temp;
    
    if (*idum <= 0 || !iy)
    
    {	/*Initialize.*/
        
        if (-(*idum) < 1)
            
            *idum=1; /*Be sure to prevent idum = 0.*/
        
        else
            
            *idum = -(*idum);
        
        for (j=NTAB+7;j>=0;j--)
        
        {	/*Load the shuffle table (after 8 warm-ups).*/
            
            k=(*idum)/IQ;
            
            *idum=IA*(*idum-k*IQ)-IR*k;
            
            if (*idum < 0) *idum += IM;
            
            if (j < NTAB) iv[j] = *idum;
            
        }
        
        iy=iv[0];
        
    }
    
    k=(*idum)/IQ;	/*Start here when not initializing.*/
    
    *idum=IA*(*idum-k*IQ)-IR*k; /*Compute idum=(IA*idum) % IM without overlows by Schrage's method.*/
    
    if (*idum < 0)
        
        *idum += IM;
    
    j=iy/NDIV;	/*Will be in the range 0..NTAB-1.*/
    
    iy=iv[j];	/*Output previously stored value and refill the	shuffle table.*/
    
    iv[j] = *idum;
    
    if ((temp=AM*iy) > RNMX)
        
        return RNMX;	/* Because users don't expect endpoint values.*/
    
    else
        
        return temp;
    
}

void print(vector<double> vv1, vector<double> vv2, char filename[])
{
    FILE * ff;
    int size;
    if (vv1.size() >= vv2.size())
    {
        size = vv2.size();
    }
    else
    {
        size = vv1.size();
    }
    ff = fopen(filename, "w");
    for (int i = 0; i < size; i++)
    {
        fprintf(ff, "%d	%f	%f\n", i, vv1[i], vv2[i]);
    }
    fclose(ff);
}
