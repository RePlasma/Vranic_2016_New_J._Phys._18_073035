"""
scattMC: monte carlo simulator of nCS + Boris pusher
"""

import numpy as np
from numpy import pi, log, log10, sqrt, sin, cos, exp
from numpy.linalg import norm
from numpy import cross, dot, floor
import matplotlib.pyplot as plt
from scipy.constants import hbar, c, m_e, alpha, e
from tqdm.notebook import tqdm
from tqdm import trange

omega_p0 = 1.88e15; # laser frequency rad s-1 (1 micron laser)
cl = c
me = m_e

def poly_sync_QED(eta, chi):
    """
    %-------------------------------------------------------------------------------
    %  function polynomial expansion for synchrontron cumulative probability for y<<1
    %  			where y = 2*chi/(3*eta(eta-2*chi))
    %-------------------------------------------------------------------------------
    """
    a = 0.921;
    b = 0.307;
    d = 2.0*chi/eta;
    I12 = 6.0 - d + (1.0/3.0)* (d**2.0) + 0.069023569023569 * (d**3.0);
    I34 =  6.0 - d + (1.0/3.0)* (d**2.0) + 0.072016460905350 * (d**3.0);
    res = 0.764540211031963 * (eta**(-1.0/3.0)) * (d**(1.0/3.0)) *( a * I12 + b * I34 );
    return res

def syncQED(eta):
    """
    %-------------------------------------------------------------------------------
    % generates a random number distributed according to the QED spectrum
    %-------------------------------------------------------------------------------
    """
    chi0 = 0.214 * eta**2 / (1 + eta**4) + (0.5*eta - 1e-6) / (1 / (eta**3) + 1);
    chimax = 0.5*eta - 1e-6;
    logchimin = log10(chi0)-12.;
    logchimax = log10(chimax);
    x = 1.0;
    P = 0.0;

    while (x > P):
        logchi = logchimin + (logchimax - logchimin)*np.random.random();
        chi = exp(log(10)*logchi);
        x = np.random.random()*peak_QED(eta);
        P = F_QED(eta,chi);
    return chi

def interpolate_photon_midrange(eta, ratio):
    """
    %-------------------------------------------------------------------------------
    %  Find chi for midrange photons 0.01 < eta < 3.3
    %-------------------------------------------------------------------------------
    """
    p1_array = [ 0.001773836232853,     0.004159773105190,     0.005909583673540,     0.006022338374199,     0.012028184722626,  \
        -0.004317070831837,     0.009330893070144,     0.022133372122301,     0.015757086764085,     0.013817061729165,   \
        -0.980486796007142,     -0.682697620064631,     -0.161298903150042,     -0.005809661704548,     0.012866929690746,  \
        -0.000482475937002,     -0.000985616750807,     0.006209647636546,     0.013027947138266,     0.019074621074484,  \
        0.001265577345342,     0.013336749438294,     0.048791963835746,     0.067308775582128,     0.044043687269626,   \
        -0.000055358248880,     -0.001348721803424,     -0.012275052087300,     -0.045221323100608,     -0.033430575699799 ];
    p2_array = [ -0.000170639654639,     0.007170182865574,     0.029058662140918,     0.034982833305507,     0.016674896115385,  \
        0.268338782993809,     0.469351068095490,     0.351483214319252,     0.141973073260982,     0.030569503524505,   \
        -4.909508871257551,     -2.996785964714983,     -0.507820495166353,     0.048413754117628,     0.026789393408900,   \
        -0.004329396889427,     -0.024295168496621,     -0.033052751294962,     -0.017905509970837,     0.018581438999043,   \
        0.004381022575082,     0.048108792452574,     0.186206593872071,     0.268302058587707,     0.154364638546128,  \
        -0.000148670396644,     -0.003846576613533,     -0.038205552382314,     -0.164785712598556,     -0.161041306668578 ];
    p3_array = [ 0.018175348373514,     0.072438926956503,     0.105531563563381,     0.067040495583811,     -0.170000259566593,   \
        0.223262330890935,     0.471684923546082,     0.388922426066648,     0.153796985221281,     -0.160391408919875,   \
        -31.680084424698883,     -20.084722542030761,  	     -4.510229318398666,     -0.359559263573851,     -0.180449508408224,   \
        -0.012347312978628,     -0.090172836468116,     -0.213250060611825,     -0.192351240589369,    -0.229109462035024,   \
        0.003111666251559,     0.039690828311627,     0.185564686028165,     0.338027042839089,     0.028759127464761,    \
        0.000023481513433,     0.000022801554947,     -0.007165145535375,     -0.081707990054295,     -0.316879491793383 ];
    p4_array = [ -0.031911342972680,     -0.144318629353999,     -0.245920478028274,     -0.190427282615784,     1.421796540765927,   \
        -2.545220550191603,     -3.553366289404980,     -2.017380580443948,     -0.610503032302649,     1.383423724377177,  \
        -109.316616098549332,     -72.579453825704149,     -18.597839190749774,     -2.373170424438470,     1.312928625714792,   \
        -0.015073626703808,     -0.125181013600805,     -0.346874427727219,     -0.241763740356303,     1.383511050857337,    \
        -0.002791369594754,     -0.022584555300707,     -0.032865372915183,     0.175959378659173,     1.587766611263539,   \
        0.000269829602639,     0.006650538450129,     0.064272326168450,     0.299244401499319,     1.624536255589060 ];
    p5_array = [ 0.049421028623077,     0.245823831180409,     0.453711481196624,     3.371207968157663,     -0.715551819439772,  \
        6.515531213872445,     9.740563653619228,     5.966701464021353,     4.890315707496041,     -0.547768232060367,   \
        93.612358800233736,     68.274849458436137,     20.721252596432443,     6.552637908179546,     -0.476703814260746,  \
        -0.038414492310977,     -0.361223808578362,     -1.312294486900120,     -2.265078646565482,     -2.015389446419847,   \
        -0.011376879355673,     -0.133755830526832,     -0.611451162601370,     -1.328331100768032,     -1.556871006962085,   \
        -0.000243937957341,     -0.005785523688170,     -0.052781127864644,     -0.226485066799127,     -0.725872671153799 ];

    lt_eta = log10(eta);

    if ( ratio < 0.4 ):
        n = 1-1;
        ltr = log10(ratio);
    elif ( ratio < 0.7 ):
        n = 6-1;
        ltr = log10(ratio);
    elif ( ratio < 0.88 ):
        n = 11-1;
        ltr = log10(ratio);
    elif (ratio < 0.975):
        n = 16-1;
        ltr = log10(1.0-ratio);
    elif (ratio < 0.9993):
        n = 21-1;
        ltr = log10(1.0-ratio);
    else:
        n = 26-1;
        ltr = log10(1.0-ratio);

    pc1 = p1_array[n] * ( lt_eta**4.0 ) + p2_array[n] * ( lt_eta**3.0 ) + p3_array[n] * ( lt_eta**2.0 ) + p4_array[n] * lt_eta + p5_array[n];
    pc2 = p1_array[n+1] * ( lt_eta**4.0 ) + p2_array[n+1] * ( lt_eta**3.0 ) + p3_array[n+1] * ( lt_eta**2.0 ) + p4_array[n+1] * lt_eta + p5_array[n+1];
    pc3 = p1_array[n+2] * ( lt_eta**4.0 ) + p2_array[n+2] * ( lt_eta**3.0 ) + p3_array[n+2] * ( lt_eta**2.0 ) + p4_array[n+2] * lt_eta + p5_array[n+2];
    pc4 = p1_array[n+3] * ( lt_eta**4.0 ) + p2_array[n+3] * ( lt_eta**3.0 ) + p3_array[n+3] * ( lt_eta**2.0 ) + p4_array[n+3] * lt_eta + p5_array[n+3];
    pc5 = p1_array[n+4] * ( lt_eta**4.0 ) + p2_array[n+4] * ( lt_eta**3.0 ) + p3_array[n+4] * ( lt_eta**2.0 ) + p4_array[n+4] * lt_eta + p5_array[n+4];
    pc = pc1 * ( ltr**4.0 ) + pc2 * ( ltr**3.0 ) + pc3 * ( ltr**2.0 ) + pc4 * ltr + pc5;
    chi = 10**pc;

    return chi


def interpolate_photon_low(eta, ratio):
    """
    %-------------------------------------------------------------------------------
    %  Find chi for low photons  eta < 0.01
    %-------------------------------------------------------------------------------
    """

    p1_array = [ 0.000263811201959,     0.000553145340450,     0.000095369097674,     -0.000411875299010,     -0.001853132487289,  \
        -0.034968036509529,     -0.048613728699486,     -0.027108238057496,     -0.007689397564446,     -0.002662591941810,  \
        -0.471389071640141,     -0.366882206365127,     -0.112218758479580,     -0.017612011812319,     -0.003089636664014,  \
        -0.000026474752394,     -0.000220282443738,     -0.000682523450780,     0.000433200917317,     -0.001573728870609,   \
        -0.000005530223021,     -0.000071572557034,     -0.000306087538604,     0.000833871524539,     -0.001422824476591,   \
        -0.000000140633974,     -0.000004012965775,     0.000014133707467,     0.001516930650954,     -0.000868627294210 ];
    p2_array = [ 0.001759803729180,     0.002219230270537,     -0.003710995832019,     -0.007443369492250,     -0.025954464715172,   \
        -0.481290048732986,     -0.671043559591993,     -0.375418889566618,     -0.106850616220667,     -0.037061410137816,   \
        -7.275620737513042,     -5.506669738559794,     -1.643778940803329,     -0.252565594681990,     -0.043269949679557,   \
        -0.000356276154489,     -0.003002703096092,     -0.009471182536737,     0.006046013708690,     -0.021917477241251,    \
        -0.000082444971341,     -0.001046577943293,     -0.004492265877667,     0.011365874881429,     -0.019913641794140,   \
        -0.000002547037527,     -0.000068162145320,     0.000053060484331,     0.020896093063842,     -0.012292213053433 ];
    p3_array = [ 0.006986723640187,     0.009997910101316,     -0.014612961168528,     -0.034208021629582,     -0.135324497875304,   \
        -2.503213435436670,     -3.500587251076293,     -1.964980868889572,     -0.561163702345697,     -0.194881296072327,  \
        -42.093636131306923,     -31.107872464612527,     -9.085101831836093,     -1.368415148610610,     -0.228959888084581,  \
        -0.001824697740166,     -0.015564332548708,     -0.049964958271390,     0.031574604130121,     -0.115395098457585,    \
        -0.000456844928734,     -0.005715449721112,     -0.024696987365157,     0.058773043081745,     -0.105091818810049,    \
        -0.000016740343327,     -0.000428850551023,     -0.000563308599435,     0.108577421492149,     -0.065824907236507 ];
    p4_array = [ 0.005335131360358,     -0.022501648387618,     -0.110725979737021,     -0.136216820923786,     1.666869897886332,   \
        -5.855109866397301,     -8.212834724838951,     -4.625499915752781,     -1.325329840321259,     1.539440714165051,   \
        -108.585158882810205,     -78.638367024033329,     -22.526356253671199,     -3.331216319051826,     1.455465999810217,  \
        -0.004237355165717,     -0.036525658976146,     -0.119237916752667,     0.073381346509617,     1.726770340767212,   \
        -0.001122548215616,     -0.013891027040214,     -0.060598648484234,     0.137146528360352,     1.751177760085899,  \
        -0.000048009457794,     -0.001192614380272,     -0.003528934286669,     0.253157886207936,     1.841329935045384 ];
    p5_array = [ 0.012440512264447,     0.058208450503953,     0.135433713807426,     3.159377819076660,     -0.576999863431748,  \
        5.328981984837045,     8.165820883577529,     5.169630202193487,     4.713848412633423,     -0.375348777377691,  \
        109.908794069604866,     75.340162347559087,     21.372576915740748,     6.468554084565493,     -0.302616117826916,  \
        -0.034560041731897,     -0.326548751981733,     -1.219198942872906,     -2.148249186408203,     -1.782748307370904,  \
        -0.009402387169714,     -0.115966993497007,     -0.573762433354587,     -1.290381599174277,     -1.365467788910251,  \
        -0.000418029920662,     -0.010630206487499,     -0.103007290874155,     -0.336591670303331,     -0.624414723674595 ];

    lt_eta = log10(eta);

    if ( ratio < 0.4 ):
        n=1-1;
        ltr = log10(ratio);
    elif ( ratio < 0.7 ):
        n = 6-1;
        ltr = log10(ratio);
    elif ( ratio < 0.88 ):
        n = 11-1;
        ltr = log10(ratio);
    elif (ratio < 0.975):
        n = 16-1;
        ltr = log10(1.0-ratio);
    elif (ratio < 0.9993):
        n = 21-1;
        ltr = log10(1.0-ratio);
    else:
        n = 26-1;
        ltr = log10(1.0-ratio);

    pc1 = p1_array[n] * ( lt_eta**4.0 ) + p2_array[n] * ( lt_eta**3.0 ) + p3_array[n] * ( lt_eta**2.0 ) + p4_array[n] * lt_eta + p5_array[n];
    pc2 = p1_array[n+1] * ( lt_eta**4.0 ) + p2_array[n+1] * ( lt_eta**3.0 ) + p3_array[n+1] * ( lt_eta**2.0 ) + p4_array[n+1] * lt_eta + p5_array[n+1];
    pc3 = p1_array[n+2] * ( lt_eta**4.0 ) + p2_array[n+2] * ( lt_eta**3.0 ) + p3_array[n+2] * ( lt_eta**2.0 ) + p4_array[n+2] * lt_eta + p5_array[n+2];
    pc4 = p1_array[n+3] * ( lt_eta**4.0 ) + p2_array[n+3] * ( lt_eta**3.0 ) + p3_array[n+3] * ( lt_eta**2.0 ) + p4_array[n+3] * lt_eta + p5_array[n+3];
    pc5 = p1_array[n+4] * ( lt_eta**4.0 ) + p2_array[n+4] * ( lt_eta**3.0 ) + p3_array[n+4] * ( lt_eta**2.0 ) + p4_array[n+4] * lt_eta + p5_array[n+4];
    pc = pc1 * ( ltr**4.0 ) + pc2 * ( ltr**3.0 ) + pc3 * ( ltr**2.0 ) + pc4 * ltr + pc5;
    chi = 10**pc;
    return chi

def interpolate_photon_high(eta, ratio):
    """
    %-------------------------------------------------------------------------------
    %  set of functions to get new chi from eta and one random value;
    %  There are 3 functions: eta> 3.3 (high), 0.01< eta < 3.3 (midrange) and eta < 0.01 (low);
    %-------------------------------------------------------------------------------
    """

    p1_array = [ 0.000394589503051,     0.001758682038381,     0.002893874817675,     0.002104730355267,     -0.003634049745981,  \
        -0.055937387233642,     -0.065886320279835,     -0.026147110013602,     -0.002940621306168,     -0.003894502232776,  \
        -1.048747279807481,     -0.872789273537110,     -0.267400248879477,     -0.034452832734123,     -0.005414601503678,   \
        1875.071802019952656,     300.495149184593060,     16.762405490950311,     0.373442696138872,     -0.001652051103320 ];
    p2_array = [ -0.003971150489006,     -0.018055330405033,     -0.030513120343084,     -0.023030679161369,     0.042273956423612, \
        0.329951816637183,     0.370401763115319,     0.126156506804666,     0.000407195758592,     0.042908607010348,   \
        -0.354179154844088,     1.286949411802133,     0.739449801699535,     0.114599218464971,     0.049666212297222,   \
        -16743.456812191390782,     -2690.227130010071960,     -151.629508314305980,     -3.554314465956319,     0.015338323224990 ];
    p3_array = [ 0.015624480455891,     0.072569603108912,     0.126108887187262,     0.098804713961978,     -0.196613435156459,  \
        -0.092237867707433,     0.054258855096189,     0.207495688575473,     0.146153041062157,     -0.189146562438188,  \
        51.069523904044814,     29.535660711749539,     6.237888502630087,     0.654622916464811,     -0.174825363262565,   \
        56887.183787550107809,     9184.043012321235437,     526.266972325473603,     13.271756641941396,     -0.054733202346443 ];
    p4_array = [ -0.029199070329492,     -0.138548698396030,     -0.247451431218712,     -0.200886031023574,     1.438475028852042,   \
        -2.638987961872001,     -3.620798753363849,     -2.020650203097647,     -0.611879714396155,     1.401780723462706,  \
        -195.122632154715916,     -125.818836023634518,     -30.690761393156599,     -3.569844737674069,     1.287758082995097,  \
        -89508.217916543173487,     -14572.859228520872421,     -856.519716321191368,     -23.819455788619784,     1.091395618969734 ];
    p5_array = [ 0.048890836069625,     0.244727685140817,     0.454442708483244,     3.374002199572533,     -0.719404122634595,   \
        6.615523470324536,     9.858398511739829,     6.018148533542798,     4.902198980643337,     -0.551056458816563,   \
        117.323516878384979,     83.233549894578488,     24.205231838998493,     6.911554027078656,     -0.466984430164604,   \
        55498.371969807310961,     9008.148552239785204,     526.826181903363249,     18.719241080848093,     -0.363030323053338 ];

    lt_eta = log10(eta);
    ltr = log10(ratio);

    if ( ratio < 0.4 ):
        n = 1-1;
    elif ( ratio < 0.7 ):
        n = 6-1;
    elif ( ratio < 0.9 ):
        n = 11-1;
    else:
        n = 16-1;

    pc1 = p1_array[n] * ( lt_eta**4.0 ) + p2_array[n] * ( lt_eta**3.0 ) + p3_array[n] * ( lt_eta**2.0 ) + p4_array[n] * lt_eta + p5_array[n];
    pc2 = p1_array[n+1] * ( lt_eta**4.0 ) + p2_array[n+1] * ( lt_eta**3.0 ) + p3_array[n+1] * ( lt_eta**2.0 ) + p4_array[n+1] * lt_eta + p5_array[n+1];
    pc3 = p1_array[n+2] * ( lt_eta**4.0 ) + p2_array[n+2] * ( lt_eta**3.0 ) + p3_array[n+2] * ( lt_eta**2.0 ) + p4_array[n+2] * lt_eta + p5_array[n+2];
    pc4 = p1_array[n+3] * ( lt_eta**4.0 ) + p2_array[n+3] * ( lt_eta**3.0 ) + p3_array[n+3] * ( lt_eta**2.0 ) + p4_array[n+3] * lt_eta + p5_array[n+3];
    pc5 = p1_array[n+4] * ( lt_eta**4.0 ) + p2_array[n+4] * ( lt_eta**3.0 ) + p3_array[n+4] * ( lt_eta**2.0 ) + p4_array[n+4] * lt_eta + p5_array[n+4];
    pc = pc1 * ( ltr**4.0 ) + pc2 * ( ltr**3.0 ) + pc3 * ( ltr**2.0 ) + pc4 * ltr + pc5;
    chi = 10**pc;
    return chi

def interp_spec_qed(eta):

    eta_array = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 0.5e0, 1.0e0, 2.0e0, 5.0e0, 1.0e1, 15.0e0, 20.0e0, \
        30.0e0, 40.0e0, 60.0e0, 80.0e0, 100.0e0, 200.0e0, 300.0e0];
    int_eval_array = [5.236e0, 5.231e0, 5.19e0, 4.87e0, 4.177e0, 3.75e0, 3.284e0, 2.659e0, 2.217e0, \
        1.98e0, 1.82e0, 1.61e0, 1.48e0, 1.3e0, 1.19e0, 1.11e0, 0.89e0, 0.78e0];

    if (eta <= 1.0e-4):
        Wrad = 5.236e0;
    else:
        loc = 17;
        for i in range(17):
            if (eta <= eta_array[i+1] and eta >= eta_array[i]):
                loc = i;
                break
                
        coef = (int_eval_array[loc+1]-int_eval_array[loc])/(eta_array[loc+1]-eta_array[loc]);
        Wrad = int_eval_array[loc] + coef*(eta-eta_array[loc]);
    return Wrad


def integral_alpha_QED(eta, chi):
    """
    %-------------------------------------------------------------------------------
    %  function estimates the integral based on improved polynomial expansion;
    %             The output is normalized to the integral from 0 to eta/2
    %-------------------------------------------------------------------------------
    """

    total = interp_spec_qed(eta);
    # this is the chi for which y = 1
    chi_y1 = 3.0 * eta * eta / ( 2.0*(1 + 3.0 * eta) );
    if ( chi > chi_y1 ):
        res = 1.0;
    else:
        delta = poly_sync_QED(eta, chi_y1) - total;
        res = ( poly_sync_QED(eta, chi) - (chi * delta / chi_y1) ) / total;
    return res

def find_photon_chi(eta):
    a = 0.921;
    b = 0.307;
    total = interp_spec_qed(eta);
    varand = np.random.random()

    # inverse from the non-improved polynomial expansion
    first_guess = 0.5*(eta**2) * ( varand * total /( 0.764540211031963 * (a + b) *6.0 ))**3;
    intt = integral_alpha_QED(eta, first_guess);
    delta_y = varand - intt;

    # this is the chi for which y = 1
    chi_y1 = 3.0 * eta * eta / ( 2.0*(1 + 3.0 * eta) );

    # if chi << chi_y1, then we can use this expansion; otherwise, we use interpolation
    if (first_guess < 0.001 * chi_y1):
        d = 2.0 * first_guess / eta;
        derivative = 0.764540211031963*(a+b)* (eta**(-4.0/3.0))* 2.0*(2.0* (d**(-2.0/3.0))-(4.0/3.0)*(d**(1.0/3.0))) /total;
        # this derivative is calculated by = (2/pi) * 3**(1/6) * (a+b) * 2 * eta**(-4/3)* (2 d**(-2/3)- (4/3)d**(1/3) )
        chi = first_guess + delta_y / derivative;
    else:
        if (eta > 3.3):
            chi_new = interpolate_photon_high(eta, varand);
        elif (eta > 0.01):
            chi_new = interpolate_photon_mid(eta, varand);
        else:
            chi_new = interpolate_photon_low(eta, varand);
        chi = chi_new;
    return res

def find_photon_chi(eta):
    a = 0.921;
    b = 0.307;
    total = interp_spec_qed(eta);
    varand = np.random.random()

    # inverse from the non-improved polynomial expansion
    first_guess =0.5*(eta**2) * ( varand * total /( 0.764540211031963 * (a + b) *6.0 ))**3;
    intt = integral_alpha_QED(eta, first_guess);
    delta_y = varand - intt;

    # this is the chi for which y = 1
    chi_y1 = 3.0 * eta * eta / ( 2.0*(1 + 3.0 * eta) );

    # if chi << chi_y1, then we can use this expansion; otherwise, we use interpolation
    if (first_guess < 0.001 * chi_y1):
        d = 2.0 * first_guess / eta;
        derivative = 0.764540211031963*(a+b)* (eta**(-4.0/3.0))* 2.0*(2.0* (d**(-2.0/3.0))-(4.0/3.0)*(d**(1.0/3.0))) /total;
        # this derivative is calculated by = (2/pi) * 3**(1/6) * (a+b) * 2 * eta**(-4/3)* (2 d**(-2/3)- (4/3)d**(1/3) )
        chi = first_guess + delta_y / derivative;
    else:
        if (eta > 3.3):
            chi_new = interpolate_photon_high(eta, varand);
        elif (eta > 0.01):
            chi_new = interpolate_photon_midrange(eta, varand);
        else:
            chi_new = interpolate_photon_low(eta, varand);
        chi = chi_new;
    return chi

def envelope(tt,trise,tflat):
    """
    envelope function (polynomial)
    """
    if (tt<0):
        res = 0;
    elif (tt<trise):
        t = tt/trise;
        res = t**3*(10-15*t+6*t**2);
    elif (tt<trise+tflat):
        res = 1;
    elif (tt<2*trise+tflat):
        t = (tt-tflat)/trise;
        res = - (-2+t)**3*(4-9*t+6*t**2);
    else:
        res = 0;
    return res

def getFields(x,t,lbd,a0,trise,tflat,tnot):
    efld=0
    
    efld = a0 * envelope(t-tnot,trise,tflat) * sin(2*pi/lbd * x);
    Evec = np.array([0, efld, 0]);
    Bvec = np.array([0, 0, -efld]);

    return Evec, Bvec

def pusher(rn,um12,t,dt,q,lbd,a0,trise,tflat,tnot):
    """
    % um12 -> um -> upr -> up -> up12
    % n-1/2 -> - -> ' -> + -> n+1/2
    """

    # Evec, Bvec
    Evec,Bvec = getFields(rn[0],t,lbd,a0,trise,tflat,tnot);

    # u- = un-1/2 + q E dt/2
    um = um12 + q*Evec*dt/2;

    # gn
    gn = sqrt(1+norm(um12)**2);

    # t
    tvec = q*Bvec*dt/(2*gn);

    # u' = u- + u- x t
    upr = um + cross(um,tvec);

    # s
    svec = 2*tvec/(1+norm(tvec)**2);

    # u+ = u- + u' x s
    up = um + cross(upr,svec);

    # un+1/2 = u+ + q E dt/2
    up12 = up + q*Evec*dt/2;

    #
    u = up12;
    v = u/sqrt(1+norm(u)**2);
    r = rn + dt*v;

    return r,u

def evolve(u00,lbd,a0,trise,tflat,tnot,dt,tdim):
    # charge
    q = -1;

    # initial position
    r = np.array([0,0,0]);
    u0 = np.array([u00,0,0]);

    # push velocity half step backwards
    _, un12 = pusher(r,u0,0,-dt/2,q,lbd,a0,trise,tflat,tnot);

    # initial momentum n-1/2
    u = un12;

    # time evolve
    t=0;
    etacount = 0;
    etadim = 1000;
    
    for n in range(tdim):

        # Boris
        r,u = pusher(r,u,t,dt,q,lbd,a0,trise,tflat,tnot);

        # Schwinger field
        # me**2 * cl**3 / (hbar*e) = 1.3233e+18
        # get fields
        Evec,Bvec = getFields(r[0],t,lbd,a0,trise,tflat,tnot);
        Ex, Ey, Ez = Evec;
        Bx, By, Bz = Bvec;
        ux, uy, uz = u;
        # calculate the quantum coefficient
        norm_schw = hbar*omega_p0/(me*cl**2);
        coef_QED = sqrt(3.0)*alpha/(2*pi*norm_schw);
        # compute lorentz factor
        p2 = norm(u)**2;
        gl = sqrt(1+p2);
        # compute eta parameter terms
        # (p · E)**2
        pdotE2 = dot(u,Evec)**2;
        # \gamma E + p x B
        gamE_plus_pcrossB2 = (-gl*Ex-Bz*uy+By*uz)**2 + (-gl*Ey+Bz*ux-Bx*uz)**2 + (-gl*Ez-By*ux+Bx*uy)**2 ;
        # compute the quantum parameter eta
        eta = sqrt(np.abs(gamE_plus_pcrossB2-pdotE2))*norm_schw;
        
        # emit photon?
        Wrad = 0;
        if ( (eta > 1.0e-4) and (eta < 5.0e4) ):
            if (eta > 300):
                Wrad = (1.46e0*alpha/norm_schw)*(eta**(2/3))/gl;
            else:
                interp = interp_spec_qed(eta);
                Wrad = interp*coef_QED*eta/gl;
            varand = np.random.random()
            # emit photon?
            if ( varand < Wrad * dt ):
                # compute photon chi
                chi_g = find_photon_chi(eta);
                # compute momentum fraction lost to emitted photon
                p_frac = 2.0e0*chi_g/eta;
                u = u*(1.0 - p_frac);

        # increment time
        t = t+dt;

    # final particle energy
    enefin = sqrt(1+norm(u)**2);

    return enefin
