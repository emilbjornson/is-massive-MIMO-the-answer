%This Matlab script can be used to generate all the simulation figures in
%the article:
%
%Emil Björnson, Luca Sanguinetti, Jakob Hoydis, Mérouane Debbah, "Optimal
%Design of Energy-Efficient Multi-User MIMO Systems: Is Massive MIMO the
%Answer?," Submitted to IEEE Transactions on Wireless Communications, March
%2014.
%
%Download article: http://arxiv.org/pdf/1403.6150
%
%This is version 1.0 (Last edited: 2014-03-21)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above.
%
%Please note that the channels are generated randomly, thus the results
%will not be exactly the same as in the paper.

%Initialization
close all;
clear all;


%%Simulation parameters

rng('shuffle'); %Initiate the random number generators with a random seed
%%If rng('shuffle'); is not supported by your Matlab version, you can use
%%the following commands instead:
%randn('state',sum(100*clock));

%Ranges of optimization parameters
Mmax = 220; %Consider all number of BS antennas from 1 to 220 in simulation
Kmax = 150; %Consider all number of active UEs from 1 to 150 in simulation

%Should Monte Carlo simulations be used for MRT/MRC processing? (true or false)
runMonteCarloSimulationsMRT = true;

%Should Monte Carlo simulations be used for MMSE processing? (true or false)
%Beware: This part of the simulation is extremly slow, due to the iterative
%power allocation algorithm! It takes weeks to finish.
runMonteCarloSimulationsMMSE = false;

%Should the sequential/alternating optimization algorithm from Section V.E
%be used? (true or false). This option should only be turned on if the
%global optimum is within the range 1,...,Mmax and 1,...,Kmax, otherwise
%there will be an error.
runSequentialAlgorithm = true;


%Geometric scenarios (From Table 2)
d_max = 250; %Cell radius for a circular cell in the single-cell scenario. It is also the distance from BS to a side of the square cells in the multi-cell scenario
d_min = 35; %Minimum distance between UE and BS
areaSinglecell = pi*(d_max/1000).^2; %Coverage area in km^2
areaMulticell = 4*(d_max/1000).^2; %Coverage area in km^2

%Large-scale fading parameters (From Table 2)
dbar = 10^(-3.53);  %Regulates channel attenuation at minimum distance (see Example 1)
kappa = 3.76; %Path-loss exponent (see Example 1)

%Spectral resources and properties (From Table 2)
B = 20e6; %Transmission bandwidth (Hz)
Bc = 180e3; %Channel coherence bandwidth (Hz)
Tc = 10e-3; %Channel coherence time (s)
U = Bc * Tc; %Coherence block (number of channel uses)
sigma2B = 10^(-9.6-3); %Total noise power (B*sigma2 in W)

%Traffic assumptions (From Table 2)
zetaDL = 0.6; %Fraction of downlink transmission
zetaUL = 0.4; %Fraction of uplink transmission

%Relative lengths of pilot sequences (From Table 2)
tauDL = 1; %Relative pilot length in the downlink
tauUL = 1; %Relative pilot length in the uplink

%Hardware characterization (From Table 2)
etaDL = 0.39; %PA efficiency at the BSs
etaUL = 0.3; %PA efficiency at the UEs
L_BS = 12.8e9; %Computational efficiency at BSs (flops/W)
L_UE = 5e9; %Computational efficiency at UEs (flops/W)
P_FIX = 18; %Fixed power consumption (control signals, backhaul, etc.) (W)
P_SYN = 2; %Power consumed by local oscillator at a BS (W)
P_BS = 1; %Power required to run the circuit components at a BS (W)
P_UE = 0.1; %Power required to run the circuit components at a UE (W)
P_COD = 0.1e-9; %Power required for channel coding (W/(bit/s))
P_DEC = 0.8e-9; %Power required for channel decoding (W/(bit/s))
P_BT = 0.25e-9; %Power required for backhaul traffic (W/(bit/s))

%System parameters computed from the parameters defined above
eta = 1/(zetaDL/etaDL + zetaUL/etaUL); %Effective PA efficiency, averaged over uplink and downlink. Defined in Eq. (19)
S_x = (d_max^(kappa+2)-d_min^(kappa+2))/dbar/(1+kappa/2)/(d_max^2-d_min^2); %Average inverse channel attenuation in the single-cell scenario (see Eq. (3))

Bsigma2SxetaSinglecell = sigma2B*S_x/eta; %Precomputation of B*sigma^2*S_x/eta for the single-cell scenario. This term appears in Eq. (19) and other places that defines the total RF power
Bsigma2SxetaMulticell = 1.602212311888643; %Value for B*sigma^2*S_x/eta in the multi-cell scenario, computed numerically. This term appears in Eq. (19) and other places that defines the total RF power

%Average relative channel attentuations in the multi-cell scenario with
%different pilot reuse factors. These numbers have been computed
%numerically.
%
%Relative channel attentuations from the interfering cells that use the
%same pilot sequences (see Section VI).
Ijl_PC_Reuse1 = [0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823];
Ijl_PC_Reuse2 = [0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823];
Ijl_PC_Reuse4 = [0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00429275445589164; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823; 0.00106280112820823];
%
%Relative channel attentuations from the interfering cells that use other
%orthogonal pilot sequences (see Section VI).
Ijl_nonPC_Reuse1 = [];
Ijl_nonPC_Reuse2 = [0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105];
Ijl_nonPC_Reuse4 = [0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0975852990937329; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.0237235488511305; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105; 0.00276741832925105];

%Parameters defined in Table 1
A = P_COD + P_DEC + P_BT; %Parameter A in the power consumption model
C_0 = P_FIX + P_SYN; %Parameter C_0 in the power consumption model
C_1 = P_UE; %Parameter C_1 in the power consumption model
C_2 = (4*B*tauDL)/(U*L_UE); %Parameter C_2 in the power consumption model
D_0 = P_BS; %Parameter D_0 in the power consumption model

C_3_ZF = B/(3*U*L_BS); %Parameter C_3 in the power consumption model for ZF processing
D_1_ZF = B*(2+1/U)/L_BS; %Parameter D_1 in the power consumption model for ZF processing
D_2_ZF = B*(3-2*tauDL)/(U*L_BS); %Parameter D_2 in the power consumption model for ZF processing

C_3_MRT = 0; %Parameter C_3 in the power consumption model for MRT/MRC processing, computed as in Section IV
D_1_MRT = B*(2+3/U)/L_BS; %Parameter D_1 in the power consumption model for MRT/MRC processing, computed as in Section IV
D_2_MRT = B*(-2*tauDL)/(U*L_BS); %Parameter D_2 in the power consumption model for MRT/MRC processing, computed as in Section IV

Q = 3; %Number of iterations in power allocation computation with MMSE processing
C_3_MMSE = C_3_ZF*Q; %Parameter C_3 in the power consumption model for MMSE processing, computed as in Section IV
D_1_MMSE = B*(2+Q*3/U)/L_BS; %Parameter D_1 in the power consumption model for MMSE processing, computed as in Section IV
D_2_MMSE = B*(3*Q-2*tauDL)/(U*L_BS); %Parameter D_2 in the power consumption model for MMSE processing, computed as in Section IV


%Initiate the random channel and noise realizations in the case of
%Monte Carlo simulations
if runMonteCarloSimulationsMRT == true || runMonteCarloSimulationsMMSE == true
    
    %Number of realizations in the Monte Carlo simulations
    nbrOfMonteCarloRealizations = 100;
    
    %Channel and noise realizations for Monte Carlo single-cell simulation
    H = ( randn(Mmax,Kmax,nbrOfMonteCarloRealizations)+1i*randn(Mmax,Kmax,nbrOfMonteCarloRealizations)) / sqrt(2); %Normalized channel matrix
    
    %Compute random user distances from BS, assuming a uniform user
    %distribution in a circular cell with maximal distance d_max and
    %minimal distance d_min.
    userDistances = sqrt( rand(Kmax,nbrOfMonteCarloRealizations)*(d_max^2-d_min^2)+ d_min^2 );
    
    l_x =  dbar ./ userDistances.^(kappa); %Compute average pathlosses based on Eq. (2)
end


%Placeholders for the single-cell results with different processing schemes
EEoptZF = zeros(Mmax,Kmax); %Maximal EE with ZF processing for different numbers of BS antennas and UEs
alphaOptsZF = zeros(Mmax,Kmax); %The corresponding alpha-parameter with ZF processing
sumRatesZF = NaN*ones(Mmax,Kmax); %The corresponding sum rates with ZF processing
RFpowersZF = NaN*ones(Mmax,Kmax); %The corresponding total RF powers with ZF processing

EEoptMRT = zeros(Mmax,Kmax); %Maximal EE with MRT/MRC processing for different numbers of BS antennas and UEs
sumRatesMRT = NaN*ones(Mmax,Kmax); %The corresponding sum rates with MRT/MRC processing
RFpowersMRT = NaN*ones(Mmax,Kmax); %The corresponding total RF powers with MRT/MRC processing

EEoptMMSE = zeros(Mmax,Kmax); %Maximal EE with MMSE processing for different numbers of BS antennas and UEs
sumRatesMMSE = NaN*ones(Mmax,Kmax); %The corresponding sum rates with MMSE processing
RFpowersMMSE = NaN*ones(Mmax,Kmax); %The corresponding total RF powers with MMSE processing

EEoptZFimperfect = zeros(Mmax,Kmax);  %Maximal EE with ZF processing (under imperfect CSI) for different numbers of BS antennas and UEs
sumRatesZFimperfect = NaN*ones(Mmax,Kmax); %The corresponding sum rates with ZF processing (under imperfect CSI)
RFpowersZFimperfect = NaN*ones(Mmax,Kmax); %The corresponding total RF powers with ZF processing (under imperfect CSI)


%Placeholders for the multi-cell results with different processing schemes
EEoptZFMulticellReuse1 = zeros(Mmax,Kmax);  %Maximal EE with ZF processing (with imperfect CSI and pilot reuse 1) for different numbers of BS antennas and UEs
sumRatesZFMulticellReuse1 = NaN*ones(Mmax,Kmax); %The corresponding sum rates with ZF processing (with imperfect CSI and pilot reuse 1)
RFpowersZFMulticellReuse1 = NaN*ones(Mmax,Kmax); %The corresponding total RF powers with ZF processing (with imperfect CSI and pilot reuse 1)

EEoptZFMulticellReuse2 = zeros(Mmax,Kmax);  %Maximal EE with ZF processing (with imperfect CSI and pilot reuse 2) for different numbers of BS antennas and UEs
sumRatesZFMulticellReuse2 = NaN*ones(Mmax,Kmax); %The corresponding sum rates with ZF processing (with imperfect CSI and pilot reuse 2)
RFpowersZFMulticellReuse2 = NaN*ones(Mmax,Kmax); %The corresponding total RF powers with ZF processing (with imperfect CSI and pilot reuse 2)

EEoptZFMulticellReuse4 = zeros(Mmax,Kmax);  %Maximal EE with ZF processing (with imperfect CSI and pilot reuse 4) for different numbers of BS antennas and UEs
sumRatesZFMulticellReuse4 = NaN*ones(Mmax,Kmax); %The corresponding sum rates with ZF processing (with imperfect CSI and pilot reuse 4)
RFpowersZFMulticellReuse4 = NaN*ones(Mmax,Kmax); %The corresponding total RF powers with ZF processing (with imperfect CSI and pilot reuse 4)


%Go through all number of BS antennas in the specified range
for M = 1:Mmax
    
    %Write out the current progress
    disp(['Current number of antennas: ' num2str(M) '. Maximum: ' num2str(Mmax)]);
    
    %Go through all number of UEs in the specified range
    for K = 1:Kmax
        
        %ZF processing is only considered for M>K. Note that it not
        %possible to perform ZF-processing for M<K.
        if M>K
            
            %Compute the parameters C' and D' which are defined in Eq. (46)
            Cprim = (C_0 + C_1*K + C_2*K^2 + C_3_ZF*K^3)/K;
            Dprim = (D_0 + D_1_ZF*K + D_2_ZF*K^2)/K;
            
            %Energy-efficiency optimization in single-cell scenario with ZF
            %processing and perfect CSI.
            alphaOptsZF(M,K) = (exp(lambertw( (M-K)*(Cprim+M*Dprim)/exp(1)/Bsigma2SxetaSinglecell - 1/exp(1) ) +1 ) -1) / (M-K); %Compute optimal alpha from Theorem 3.
            RFpowersZF(M,K) = alphaOptsZF(M,K)*K*Bsigma2SxetaSinglecell; %Compute the corresponding RF power according to Eq. (19)
            sumRatesZF(M,K) = B*K*(1-(tauUL+tauDL)*K/U)*log2(1+alphaOptsZF(M,K)*(M-K)); %Compute the corresponding sum rate according to Eq. (32)
            EEoptZF(M,K) = ( sumRatesZF(M,K) ) / (  RFpowersZF(M,K) + C_0 + C_1*K + C_2*K^2 + C_3_ZF*K^3 + D_0*M + D_1_ZF*K*M + D_2_ZF*K^2*M + A*sumRatesZF(M,K) ); %Compute the corresponding EE value according to Eq. (32)
            
            
            %Energy-efficiency optimization in single-cell scenario with ZF
            %processing and imperfect CSI.
            prelogFactor = B*K*(1-(tauUL+tauDL)*K/U); %Factor in front of the logarithm in the rate expression in Eq. (32)
            circuitpowerZF = C_0 + C_1*K + C_2*K^2 + C_3_ZF*K^3 + D_0*M + D_1_ZF*K*M + D_2_ZF*K^2*M; %Compute circuit power (i.e., excluding RF power) with ZF processing and current number of antennas/UEs
            
            %Find the optimal alpha-value by numerical optimization, using
            %the rate expression derived in Lemma 5.
            alphaZFimp = fminsearch(@(x) -functionCalculateEEMulticellZF(x,M,K,[],[],tauUL,Bsigma2SxetaSinglecell,A,circuitpowerZF,prelogFactor)/1e6, alphaOptsZF(M,K)/2);
            [EEvalue,sumrateSinglecell,averageRFPower] = functionCalculateEEMulticellZF(alphaZFimp,M,K,[],[],tauUL,Bsigma2SxetaSinglecell,A,circuitpowerZF,prelogFactor);
            
            %Save the results that were obtained in the optimization
            sumRatesZFimperfect(M,K) = sumrateSinglecell;
            EEoptZFimperfect(M,K) = EEvalue;
            RFpowersZFimperfect(M,K) = averageRFPower;
            
            
            %Energy-efficiency optimization in multi-cell scenario with ZF
            %processing and imperfect CSI. Here: Pilot reuse 1.
            prelogChannelReuse1 = B*K*(1-(tauUL+tauDL)*K/U); %It is actually the same as above when the pilot reuse is 1.
            
            %Find the optimal alpha-value by numerical optimization, using
            %the rate expression derived in Lemma 6.
            alphaZFMulticell = fminsearch(@(x) -functionCalculateEEMulticellZF(x,M,K,Ijl_PC_Reuse1,Ijl_nonPC_Reuse1,tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse1)/1e6, alphaOptsZF(M,K)/2);
            [EEvalue,rateMulticell,averageRFPower] = functionCalculateEEMulticellZF(alphaZFMulticell,M,K,Ijl_PC_Reuse1,Ijl_nonPC_Reuse1,tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse1);
            
            %Save the results that were obtained in the optimization
            sumRatesZFMulticellReuse1(M,K) = rateMulticell;
            EEoptZFMulticellReuse1(M,K) = EEvalue;
            RFpowersZFMulticellReuse1(M,K) = averageRFPower;
            
            
            %Energy-efficiency optimization in multi-cell scenario with ZF
            %processing and imperfect CSI. Here: Pilot reuse 2.
            prelogChannelReuse2 = B*K*(1-(2*tauUL+tauDL)*K/U); %Note that only the UL part is multiplied with the pilot reuse factor. Moreover, note that the circuit power parameters in Table 1 are independent of tauUL.
            
            %Find the optimal alpha-value by numerical optimization, using
            %the rate expression derived in Lemma 6.
            alphaZFMulticell = fminsearch(@(x) -functionCalculateEEMulticellZF(x,M,K,Ijl_PC_Reuse2,Ijl_nonPC_Reuse2,2*tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse2)/1e6, alphaOptsZF(M,K)/2);
            [EEvalue,rateMulticell,averageRFPower] = functionCalculateEEMulticellZF(alphaZFMulticell,M,K,Ijl_PC_Reuse2,Ijl_nonPC_Reuse2,2*tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse2);
            
            %Save the results that were obtained in the optimization
            sumRatesZFMulticellReuse2(M,K) = rateMulticell;
            EEoptZFMulticellReuse2(M,K) = EEvalue;
            RFpowersZFMulticellReuse2(M,K) = averageRFPower;
            
            
            %Energy-efficiency optimization in multi-cell scenario with ZF
            %processing and imperfect CSI. Here: Pilot reuse 4.
            prelogChannelReuse4 = B*K*(1-(4*tauUL+tauDL)*K/U); %Note that only the UL part is multiplied with the pilot reuse factor. Moreover, note that the circuit power parameters in Table 1 are independent of tauUL.
            
            %Find the optimal alpha-value by numerical optimization, using
            %the rate expression derived in Lemma 6.
            alphaZFMulticell = fminsearch(@(x) -functionCalculateEEMulticellZF(x,M,K,Ijl_PC_Reuse4,Ijl_nonPC_Reuse4,4*tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse4)/1e6, alphaOptsZF(M,K)/2);
            [EEvalue,rateMulticell,averageRFPower] = functionCalculateEEMulticellZF(alphaZFMulticell,M,K,Ijl_PC_Reuse4,Ijl_nonPC_Reuse4,4*tauUL,Bsigma2SxetaMulticell,A,circuitpowerZF,prelogChannelReuse4);
            
            %Save the results that were obtained in the optimization
            sumRatesZFMulticellReuse4(M,K) = rateMulticell;
            EEoptZFMulticellReuse4(M,K) = EEvalue;
            RFpowersZFMulticellReuse4(M,K) = averageRFPower;
            
        end
        
        
        if runMonteCarloSimulationsMRT == true || runMonteCarloSimulationsMMSE == true
            
            %Placeholder for channel realizations with distance-dependent
            %channel attentuation
            Hchannel = zeros(M,K,nbrOfMonteCarloRealizations);
            
            if runMonteCarloSimulationsMRT == true
                %Placeholders for the diagonal and off-diagonal elements of
                %D^(ul) defined in Eq. (9), when using MRT/MRC processing
                DdiagMRT = zeros(K,K,nbrOfMonteCarloRealizations);
                DoffdiagMRT = zeros(K,K,nbrOfMonteCarloRealizations);
            end
            
            %Go through all Monte Carlo realizations
            for itr = 1:nbrOfMonteCarloRealizations
                
                %Current channel realization, including the distance-dependent pathloss
                Hchannel(:,:,itr) = repmat(sqrt(l_x(1:K,itr)'),[M 1]) .* H(1:M,1:K,itr);
                
                if runMonteCarloSimulationsMRT == true
                    
                    %Compute the matrix D^(ul) in Eq. (9) for MRC processing
                    G_MRC = Hchannel(:,:,itr); %Compute MRC matrix
                    G_MRC = G_MRC ./ repmat(sqrt(sum(abs(G_MRC).^2,1)),[M 1]); %Normalize each column
                    
                    Gains = abs(Hchannel(:,:,itr)'*G_MRC).^2/sigma2B; %Multiply the MRC matrix with all the current channels and normalize by noise power
                    
                    DdiagMRT(:,:,itr) = diag(diag(Gains)); %Store the diagonal elements of D^(ul) for current Monte Carlo realization (except for the factor 1/(2^(Rbar/B)-1))
                    DoffdiagMRT(:,:,itr) = diag(diag(Gains)) - Gains; %Store the off-diagonal elements of D^(ul) for current Monte Carlo realization
                    
                end
                
            end
            
            
            %Factor in front of the logarithm in the rate expression in Eq. (32)
            prelogFactor = B*K*(1-(tauUL+tauDL)*K/U);
            
            %Factor in front of the total RF power in Eq. (19)
            prefixPower = 1/eta;
            
            %Multiplicative factor of the EE (from bit/J to Mbit/J) to
            %limit the reduce of EE values in the numerical optimization
            scalingFactor = 1e6;
            
            %Initial value for the SINR in the Monte Carlo simulation
            initialSINR = 0.1;
            
            
            %Energy-efficiency optimization in single-cell scenario with
            %MRT/MRC processing and perfect CSI. Based on Monte Carlo
            %simulations
            if runMonteCarloSimulationsMRT == true
                
                %Compute circuit power (i.e., excluding RF power) with MRT/MRC processing and current number of antennas/UEs
                circuitpowerMRT = C_0 + C_1*K + C_2*K^2 + C_3_MRT*K^3 + D_0*M + D_1_MRT*K*M + D_2_MRT*K^2*M;
                
                %Find the maximal EE and corresponding SINRs using
                %numerical optimization (this one is relatively fast)
                [SINRMRT,optvalue] = fminsearch(@(x) -functionCalculateEmpircalEE(x,DdiagMRT,DoffdiagMRT,circuitpowerMRT,prelogFactor,A,prefixPower)/scalingFactor, initialSINR);
                
                %Save the results that were obtained in the optimization
                EEoptMRT(M,K) = -optvalue*scalingFactor;
                sumRatesMRT(M,K) = prelogFactor*log2(1+SINRMRT);
                RFpowersMRT(M,K) = sumRatesMRT(M,K)/EEoptMRT(M,K) - A*sumRatesMRT(M,K)-circuitpowerMRT;
                
            end
            
            
            %Energy-efficiency optimization in single-cell scenario with
            %MMSE processing and perfect CSI. Based on Monte Carlo
            %simulations
            if runMonteCarloSimulationsMMSE == true
                
                %Compute circuit power (i.e., excluding RF power) with MMSE processing and current number of antennas/UEs
                circuitpowerMMSE = C_0 + C_1*K + C_2*K^2 + C_3_MMSE*K^3 + D_0*M + D_1_MMSE*K*M + D_2_MMSE*K^2*M;
                
                %Find the maximal EE and corresponding SINRs using
                %numerical optimization (beware: this one is very slow)
                [SINRMMSE,optvalue] = fminsearch(@(x) -functionCalculateEmpircalEE_MMSE(x,Hchannel,circuitpowerMMSE,A,prelogFactor,prefixPower,sigma2B,l_x(1:K,:),Q)/scalingFactor, initialSINR);
                
                %Save the results that were obtained in the optimization
                EEoptMMSE(M,K) = -optvalue*scalingFactor;
                sumRatesMMSE(M,K) = prelogFactor*log2(1+SINRMMSE);
                RFpowersMMSE(M,K) = sumRatesMMSE(M,K)/EEoptMMSE(M,K) - A*sumRatesMMSE(M,K)-circuitpowerMMSE;
                
            end
            
        end
    end
end



%Sequential/alternating optimization algorithm from Section V.E
if runSequentialAlgorithm == true
    
    %Maximal number of iterations in the algorithm
    maximalIterationsSequential = 100;
    
    
    %Placeholders for storing the M, K, alpha, and EE at each iteration
    M_sequential = zeros(maximalIterationsSequential+1,1);
    K_sequential = zeros(maximalIterationsSequential+1,1);
    alpha_sequential = zeros(maximalIterationsSequential+1,1);
    EE_sequential = zeros(maximalIterationsSequential+1,1);
    
    %Initiate the algorithm at M=3 and K=1.
    M_sequential(1) = 3;
    K_sequential(1) = 1;
    alpha_sequential(1) = 1;
    EE_sequential(1) = EEoptZF(M_sequential(1),K_sequential(1));
    
    
    %Run the algorithm until convergence or until the maximal number of
    %iterations have been reached.
    for itrSequential = 1:maximalIterationsSequential
        
        %Step 1 in the sequential algorithm: Update K
        
        alphaBar = alpha_sequential(itrSequential)*K_sequential(itrSequential); %Fix the total transmit power
        betaBar = M_sequential(itrSequential)/K_sequential(itrSequential); %Fix the ratio between antennas and UEs
        
        cBar = B*log2(1+alphaBar*(betaBar-1)); %Compute the rate/UE achieved with fixed power and fixed antenna/UE ratio
        mu0 = (C_0 + Bsigma2SxetaSinglecell*alphaBar)/(C_3_ZF + betaBar*D_2_ZF); %Compute mu_0 as defined in Theorem 1
        mu1 = (U*(C_2+betaBar*D_1_ZF)/(tauUL+tauDL) + C_1 + betaBar*D_0)/(C_3_ZF + betaBar*D_2_ZF); %Compute mu_1 as defined in Theorem 1
        
        polynomial = [1 -2*U/(tauUL+tauDL) -mu1 -2*mu0 U*mu0/(tauUL+tauDL)]; %Compute the coefficients of the polynomial in Eq. (42)
        Kopt_values = roots(polynomial); %Compute the 4 roots of the polynomial, which gives the optimal K and some extra values
        Kopt_values = Kopt_values(Kopt_values>0); %Discard the roots that are not real-valued and positive
        
        RFpowerCurrent = alphaBar*Bsigma2SxetaSinglecell; %Compute the current RF power
        sumRatesEachK = Kopt_values.*(1-(tauUL+tauDL)*Kopt_values/U)*cBar; %Compute the current sum rate (for the each of the remaining roots)
        
        %Compute the EE achieved with each of the remaining roots
        EEvaluesEachK = ( sumRatesEachK ) ./ (  RFpowerCurrent + C_0 + C_1*Kopt_values + C_2*Kopt_values.^2 + C_3_ZF*Kopt_values.^3 + D_0*betaBar*Kopt_values + D_1_ZF*Kopt_values.^2*betaBar + D_2_ZF*Kopt_values.^3*betaBar + A*sumRatesEachK );
        [~,indM] = max(EEvaluesEachK); %Find the index of the root that maximizes the EE
        K_currentOptimal = Kopt_values(indM); %Save the K value that maximizes the EE
        alphaCurrentValue = alphaBar/K_currentOptimal; %Compute the current alpha value, based on the fixed transmit power and optimal K
        
        
        %Step 2 in the sequential algorithm: Update M
        %This optimal M value follows directly from Eq. (48) in Theorem 2
        CprimCurrentValue = (C_0 + C_1*K_currentOptimal + C_2*K_currentOptimal^2 + C_3_ZF*K_currentOptimal^3)/K_currentOptimal;
        DprimCurrentValue = (D_0 + D_1_ZF*K_currentOptimal + D_2_ZF*K_currentOptimal^2)/K_currentOptimal;
        Mopt = (exp(lambertw(alphaCurrentValue*(Bsigma2SxetaSinglecell*alphaCurrentValue+CprimCurrentValue)/DprimCurrentValue/exp(1) + (alphaCurrentValue*K_currentOptimal-1)/exp(1) )+1) + alphaCurrentValue*K_currentOptimal-1)/alphaCurrentValue;
        
        
        %Step 3 in the sequential algorithm: Update alpha
        
        %The current M and K are non-integers, but the concavity implies
        %that the optimal K and M must be among the closest integers
        integerPoints = [floor(Mopt) floor(K_currentOptimal); floor(Mopt) ceil(K_currentOptimal); ceil(Mopt) floor(K_currentOptimal); ceil(Mopt) ceil(K_currentOptimal)];
        
        %We use Theorem 3 to cmpute the optimal alpha for each of the
        %candidate M and K values. Note that we have already computed the
        %optimal alpha for every M and K in the range defined
        %by Mmax and Kmax. We therefore use these values directly.
        EEvalues = [EEoptZF(integerPoints(1,1),integerPoints(1,2)); EEoptZF(integerPoints(2,1),integerPoints(2,2)); EEoptZF(integerPoints(3,1),integerPoints(3,2)); EEoptZF(integerPoints(4,1),integerPoints(4,2))];
        
        %Pick out the EE maximal value of M and K, among the integer
        %combinations that we obtained in this iteration.
        [~,indM] = max(EEvalues);
        
        %Store the M, K, and alpha values obtained in this iteration, and
        %the corresponding EE.
        M_sequential(itrSequential+1) = integerPoints(indM,1);
        K_sequential(itrSequential+1) = integerPoints(indM,2);
        alpha_sequential(itrSequential+1) = alphaOptsZF(M_sequential(itrSequential+1),K_sequential(itrSequential+1));
        EE_sequential(itrSequential+1) = EEoptZF(M_sequential(itrSequential+1),K_sequential(itrSequential+1));
        
        %Check if we have changed M and K as compared to the previous
        %iteration. If not, the algorithm has converged and we stop it.
        if M_sequential(itrSequential+1)==M_sequential(itrSequential) && K_sequential(itrSequential+1)==K_sequential(itrSequential)
            break;
        end
        
    end
    
end




%%The rest of the script takes care of plotting the results.

%Density of the lines that are used in the 3d plots to make it easier to
%see the shape
gridDensity = 25;


%Plot Figure 3: Energy efficiency (in Mbit/Joule) with ZF processing in
%the single-cell scenario with perfect CSI.
figure(3); hold on; box on;
title('Figure 3: ZF processing, Single-cell, Perfect CSI')

surface(1:Kmax,1:Mmax,EEoptZF/1e6,'EdgeColor','none'); %Plot the 3d surface
colormap(autumn);

%Compute and plot the optimal point
[EEvalues,indM] = max(EEoptZF,[],2);
[EEoptimal,indK] = max(EEvalues);
plot3(indM(indK),indK,EEoptimal/1e6,'k*','MarkerSize',10);

if runSequentialAlgorithm == true
    plot3(K_sequential(1:itrSequential),M_sequential(1:itrSequential),EE_sequential(1:itrSequential)/1e6,'ko-');
end

%Plot lines on top of the 3d surface, to make it easier to see the shape
for m = [1 gridDensity:gridDensity:Mmax]
    plot3(1:Kmax,m*ones(1,Kmax),EEoptZF(m,:)/1e6,'k-');
end

for k = [1 gridDensity:gridDensity:Kmax]
    plot3(k*ones(1,Mmax),1:Mmax,EEoptZF(:,k)/1e6,'k-');
end

plot3(1:Kmax,1:Kmax,zeros(Kmax,1),'k-');

view([-46 24]);
axis([0 Kmax 0 Mmax 0 35]);

ylabel('Number of Antennas (M)');
xlabel('Number of Users (K)');
zlabel('Energy Efficiency [Mbit/Joule]');



if runMonteCarloSimulationsMMSE == true
    
    %Plot Figure 4: Energy efficiency (in Mbit/Joule) with MMSE processing in
    %the single-cell scenario with perfect CSI.
    figure(4); hold on; box on;
    title('Figure 4: MMSE processing, Single-cell, Perfect CSI');
    
    surface(1:Kmax,1:Mmax,EEoptMMSE/1e6,'EdgeColor','none');  %Plot the 3d surface
    colormap(autumn);
    
    %Compute and plot the optimal point
    [EEvalues,ind] = max(EEoptMMSE,[],2);
    [EEoptimal, ind2] = max(EEvalues);
    plot3(ind(ind2),ind2,EEoptimal/1e6,'k*','MarkerSize',10);
    
    %Plot lines on top of the 3d surface, to make it easier to see the shape
    for m = [1 gridDensity:gridDensity:Mmax]
        plot3(1:Kmax,m*ones(1,Kmax),EEoptMMSE(m,:)/1e6,'k-');
    end
    
    for k = [1 gridDensity:gridDensity:Kmax]
        plot3(k*ones(1,Mmax),1:Mmax,EEoptMMSE(:,k)/1e6,'k-');
    end
    
    view([-46 24]);
    axis([0 Kmax 0 Mmax 0 35]);
    
    ylabel('Number of Antennas (M)');
    xlabel('Number of Users (K)');
    zlabel('Energy Efficiency [Mbit/Joule]');
    
end



if runMonteCarloSimulationsMRT == true
    
    %Plot Figure 5: Energy efficiency (in Mbit/Joule) with MRT/MRC processing in
    %the single-cell scenario with perfect CSI.
    figure(5); hold on; box on;
    title('Figure 5: MRT/MRC processing, Single-cell, Perfect CSI');
    
    surface(1:Kmax,1:Mmax,EEoptMRT/1e6,'EdgeColor','none');  %Plot the 3d surface
    colormap(autumn);
    
    %Compute and plot the optimal point
    [EEvalues,ind] = max(EEoptMRT,[],2);
    [EEoptimal, ind2] = max(EEvalues);
    plot3(ind(ind2),ind2,EEoptimal/1e6,'k*','MarkerSize',10);
    
    %Plot lines on top of the 3d surface, to make it easier to see the shape
    for m = [1 gridDensity:gridDensity:Mmax]
        plot3(1:Kmax,m*ones(1,Kmax),EEoptMRT(m,:)/1e6,'k-');
    end
    
    for k = [1 gridDensity:gridDensity:Kmax]
        plot3(k*ones(1,Mmax),1:Mmax,EEoptMRT(:,k)/1e6,'k-');
    end
    
    view([-46 24]);
    axis([0 Kmax 0 Mmax 0 12]);
    
    ylabel('Number of Antennas (M)');
    xlabel('Number of Users (K)');
    zlabel('Energy Efficiency [Mbit/Joule]');
    
end


%Plot Figure 6: Energy efficiency (in Mbit/Joule) with ZF processing in
%the single-cell scenario with imperfect CSI.
figure(6); hold on; box on;
title('Figure 6: ZF processing, Single-cell, Imperfect CSI')

surface(1:Kmax,1:Mmax,EEoptZFimperfect/1e6,'EdgeColor','none'); %Plot the 3d surface
colormap(autumn);

%Compute and plot the optimal point
[EEvalues,indM] = max(EEoptZFimperfect,[],2);
[EEoptimal,indK] = max(EEvalues);
plot3(indM(indK),indK,EEoptimal/1e6,'k*','MarkerSize',10);

%Plot lines on top of the 3d surface, to make it easier to see the shape
for m = [1 gridDensity:gridDensity:Mmax]
    plot3(1:Kmax,m*ones(1,Kmax),EEoptZFimperfect(m,:)/1e6,'k-');
end

for k = [1 gridDensity:gridDensity:Kmax]
    plot3(k*ones(1,Mmax),1:Mmax,EEoptZFimperfect(:,k)/1e6,'k-');
end

plot3(1:Kmax,1:Kmax,zeros(Kmax,1),'k-');

view([-46 24])
axis([0 Kmax 0 Mmax 0 30])

ylabel('Number of Antennas (M)');
xlabel('Number of Users (K)');
zlabel('Energy Efficiency [Mbit/Joule]');



%Figures 7-9: Maximal EE in the single-cell scenario and the corresponding
%sum rates and total RF powers.

Mrange = (1:Mmax)'; %The range of antenna values.

%Find the EE-maximizing number of UEs for each number of antennas and each
%processing scheme.
[~,optKzf] = max(EEoptZF,[],2);
[~,optKmrt] = max(EEoptMRT,[],2);
[~,optKmmse] = max(EEoptMMSE,[],2);
[~,optKzfimperfect] = max(EEoptZFimperfect,[],2);

%Placeholders for storing the maximal EE for different number of antennas M
%and for each processing scheme.
optEEsZF = zeros(Mmax,1);
optEEsMRT = zeros(Mmax,1);
optEEsMMSE = zeros(Mmax,1);
optEEsZFimperfect = zeros(Mmax,1);

%Placeholders for storing the EE-optimal RF power for different M and for
%each processing scheme.
optRFpowersZF = zeros(Mmax,1);
optRFpowersMRT = zeros(Mmax,1);
optRFpowersMMSE = zeros(Mmax,1);
optRFpowersZFimperfect = zeros(Mmax,1);

%Placeholders for storing the EE-optimal sum rates for different M and for
%each processing scheme.
optEEsumratesZF = zeros(Mmax,1);
optEEsumratesMRT = zeros(Mmax,1);
optEEsumratesMMSE = zeros(Mmax,1);
optEEsumratesZFimperfect = zeros(Mmax,1);

%Go through all different M
for M = 1:Mmax
    
    %Store the maximal EE for different number of antennas M (normalized to Mbit/Joule)
    optEEsZF(M) = EEoptZF(M,optKzf(M))/1e6;
    optEEsMRT(M) = EEoptMRT(M,optKmrt(M))/1e6;
    optEEsMMSE(M) = EEoptMMSE(M,optKmmse(M))/1e6;
    optEEsZFimperfect(M) = EEoptZFimperfect(M,optKzfimperfect(M))/1e6;
    
    %Store the corresponding EE-optimal RF power
    optRFpowersZF(M) = RFpowersZF(M,optKzf(M));
    optRFpowersMRT(M) = RFpowersMRT(M,optKmrt(M));
    optRFpowersMMSE(M) = RFpowersMMSE(M,optKmmse(M));
    optRFpowersZFimperfect(M) = RFpowersZFimperfect(M,optKzfimperfect(M));
    
    %Store the corresponding EE-optimal sum rates (normalized to Gbit/s)
    optEEsumratesZF(M) = sumRatesZF(M,optKzf(M))/1e9;
    optEEsumratesMRT(M) = sumRatesMRT(M,optKmrt(M))/1e9;
    optEEsumratesMMSE(M) = sumRatesMMSE(M,optKmmse(M))/1e9;
    optEEsumratesZFimperfect(M) = sumRatesZFimperfect(M,optKzfimperfect(M))/1e9;
    
end

%Compute and store the value of M that maximizes the EE globally. It is
%used to draw circles at the optimal values in Figures 7-9.
[~,MoptimalZF] = max(optEEsZF);
[~,MoptimalMRT] = max(optEEsMRT);
[~,MoptimalMMSE] = max(optEEsMMSE);
[~,MoptimalZFimperfect] = max(optEEsZFimperfect);


%Plot Figure 7: Maximal EE in the single-cell scenario for different number
%of antennas.
figure(7); hold on; box on;
title('Figure 7: Single-cell, Comparison of EE values');

plot(Mrange,optEEsMMSE,'r-.','LineWidth',1);
plot(Mrange,optEEsZF,'k','LineWidth',1);
plot(Mrange,optEEsZFimperfect,'k:','LineWidth',1);
plot(Mrange,optEEsMRT,'b--','LineWidth',1);

plot(MoptimalMMSE,optEEsMMSE(MoptimalMMSE),'ro','LineWidth',1);
plot(MoptimalZF,optEEsZF(MoptimalZF),'ko','LineWidth',1);
plot(MoptimalZFimperfect,optEEsZFimperfect(MoptimalZFimperfect),'ko','LineWidth',1);
plot(MoptimalMRT,optEEsMRT(MoptimalMRT),'bo','LineWidth',1);

axis([0 Mmax 0 35]);

legend('MMSE (Perfect CSI)','ZF (Perfect CSI)','ZF (Imperfect CSI)','MRT (Perfect CSI)','Location','Best')

xlabel('Number of Antennas (M)');
ylabel('Energy Efficiency [Mbit/Joule]');


%Plot Figure 8: Total RF power at the EE-maximizing solution in the
%single-cell scenario for different number of antennas. The radiated power 
%per BS antennas is also shown.
figure(8); hold on; box on;
title('Figure 8: Single-cell, Comparison of RF power and radiated power/antenna');

plot(Mrange,optRFpowersMMSE,'r-.','LineWidth',1);
plot(Mrange,optRFpowersZF,'k','LineWidth',1);
plot(Mrange,optRFpowersZFimperfect,'k:','LineWidth',1);
plot(Mrange,optRFpowersMRT,'b--','LineWidth',1);

plot(MoptimalMMSE,optRFpowersMMSE(MoptimalMMSE),'ro','LineWidth',1);
plot(MoptimalZF,optRFpowersZF(MoptimalZF),'ko','LineWidth',1);
plot(MoptimalZFimperfect,optRFpowersZFimperfect(MoptimalZFimperfect),'ko','LineWidth',1);
plot(MoptimalMRT,optRFpowersMRT(MoptimalMRT),'bo','LineWidth',1);

plot(Mrange,zetaDL*eta*optRFpowersMMSE./Mrange,'r-.','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersZF./Mrange,'k','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersZFimperfect./Mrange,'k:','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersMRT./Mrange,'b--','LineWidth',1);

plot(MoptimalMMSE,zetaDL*eta*optRFpowersMMSE(MoptimalMMSE)./MoptimalMMSE,'ro','LineWidth',1);
plot(MoptimalZF,zetaDL*eta*optRFpowersZF(MoptimalZF)./MoptimalZF,'ko','LineWidth',1);
plot(MoptimalZFimperfect,zetaDL*eta*optRFpowersZFimperfect(MoptimalZFimperfect)./MoptimalZFimperfect,'ko','LineWidth',1);
plot(MoptimalMRT,zetaDL*eta*optRFpowersMRT(MoptimalMRT)./MoptimalMRT,'bo','LineWidth',1);

set(gca,'YScale','Log');
axis([0 Mmax 1e-2 1e2]);

text(35,4.5,'Total RF power');
text(35,0.25,'Radiated power per BS antenna');
legend('MMSE (Perfect CSI)','ZF (Perfect CSI)','ZF (Imperfect CSI)','MRT (Perfect CSI)','Location','Best')

xlabel('Number of Antennas (M)');
ylabel('Average Power [W]');



%Plot Figure 9: Area throughput at the EE-maximizing solution in the
%multi-cell scenario, for different number of BS antennas.
figure(9); hold on; box on;
title('Figure 9: Single-cell, Comparison of area throughput');

plot(Mrange,optEEsumratesMMSE/areaSinglecell,'r-.','LineWidth',1);
plot(Mrange,optEEsumratesZF/areaSinglecell,'k','LineWidth',1);
plot(Mrange,optEEsumratesZFimperfect/areaSinglecell,'k:','LineWidth',1);
plot(Mrange,optEEsumratesMRT/areaSinglecell,'b--','LineWidth',1);

plot(MoptimalMMSE,optEEsumratesMMSE(MoptimalMMSE)/areaSinglecell,'ro','LineWidth',1);
plot(MoptimalZF,optEEsumratesZF(MoptimalZF)/areaSinglecell,'ko','LineWidth',1);
plot(MoptimalZFimperfect,optEEsumratesZFimperfect(MoptimalZFimperfect)/areaSinglecell,'ko','LineWidth',1);
plot(MoptimalMRT,optEEsumratesMRT(MoptimalMRT)/areaSinglecell,'bo','LineWidth',1);

axis([0 Mmax 0 70]);

legend('MMSE (Perfect CSI)','ZF (Perfect CSI)','ZF (Imperfect CSI)','MRT (Perfect CSI)','Location','NorthWest')

xlabel('Number of Antennas (M)');
ylabel('Area Throughput [Gbit/s/km^2]');



%Figures 11-13: Maximal EE in the multi-cell scenario and the corresponding
%sum rates and total RF powers.

%Find the EE-maximizing number of UEs for each number of antennas.
[~,optKMulticellReuse1] = max(EEoptZFMulticellReuse1,[],2);
[~,optKMulticellReuse2] = max(EEoptZFMulticellReuse2,[],2);
[~,optKMulticellReuse4] = max(EEoptZFMulticellReuse4,[],2);

%Placeholders for storing the maximal EE for different number of antennas M
%and for each pilot reuse factor.
optEEsMulticellReuse1 = zeros(Mmax,1);
optEEsMulticellReuse2 = zeros(Mmax,1);
optEEsMulticellReuse4 = zeros(Mmax,1);

%Placeholders for storing the EE-optimal RF power for different M and for
%each pilot reuse factor.
optRFpowersMulticellReuse1 = zeros(Mmax,1);
optRFpowersMulticellReuse2 = zeros(Mmax,1);
optRFpowersMulticellReuse4 = zeros(Mmax,1);

%Placeholders for storing the EE-optimal sum rates for different M and for
%each pilot reuse factor.
optEEsumratesMulticellReuse1 = zeros(Mmax,1);
optEEsumratesMulticellReuse2 = zeros(Mmax,1);
optEEsumratesMulticellReuse4 = zeros(Mmax,1);

%Go through all different M
for M = 1:Mmax
    
    %Store the maximal EE for different number of antennas M (normalized to Mbit/Joule)
    optEEsMulticellReuse1(M) = EEoptZFMulticellReuse1(M,optKMulticellReuse1(M))/1e6;
    optEEsMulticellReuse2(M) = EEoptZFMulticellReuse2(M,optKMulticellReuse2(M))/1e6;
    optEEsMulticellReuse4(M) = EEoptZFMulticellReuse4(M,optKMulticellReuse4(M))/1e6;
    
    %Store the corresponding EE-optimal RF power
    optRFpowersMulticellReuse1(M) = RFpowersZFMulticellReuse1(M,optKMulticellReuse1(M));
    optRFpowersMulticellReuse2(M) = RFpowersZFMulticellReuse2(M,optKMulticellReuse2(M));
    optRFpowersMulticellReuse4(M) = RFpowersZFMulticellReuse4(M,optKMulticellReuse4(M));
    
    %Store the corresponding EE-optimal sum rates (normalized to Gbit/s)
    optEEsumratesMulticellReuse1(M) = sumRatesZFMulticellReuse1(M,optKMulticellReuse1(M))/1e9;
    optEEsumratesMulticellReuse2(M) = sumRatesZFMulticellReuse2(M,optKMulticellReuse2(M))/1e9;
    optEEsumratesMulticellReuse4(M) = sumRatesZFMulticellReuse4(M,optKMulticellReuse4(M))/1e9;
    
end

%Compute and store the value of M that maximizes the EE globally. It is
%used to draw circles at the optimal values in Figures 11-13.
[~,MoptimalReuse1] = max(optEEsMulticellReuse1);
[~,MoptimalReuse2] = max(optEEsMulticellReuse2);
[~,MoptimalReuse4] = max(optEEsMulticellReuse4);


%Plot Figure 11: Maximal EE in the multi-cell scenario for different number
%of antennas.
figure(11); hold on; box on;
title('Figure 11: Multi-cell, Comparison of EE values');

plot(Mrange,optEEsMulticellReuse4,'b--','LineWidth',1);
plot(Mrange,optEEsMulticellReuse2,'k','LineWidth',1);
plot(Mrange,optEEsMulticellReuse1,'r-.','LineWidth',1);

plot(MoptimalReuse4,optEEsMulticellReuse4(MoptimalReuse4),'bo','LineWidth',1);
plot(MoptimalReuse2,optEEsMulticellReuse2(MoptimalReuse2),'ko','LineWidth',1);
plot(MoptimalReuse1,optEEsMulticellReuse1(MoptimalReuse1),'ro','LineWidth',1);

axis([0 Mmax 0 8]);

legend('ZF (Imperfect CSI): Reuse 4','ZF (Imperfect CSI): Reuse 2','ZF (Imperfect CSI): Reuse 1','Location','SouthEast');

xlabel('Number of Antennas (M)');
ylabel('Energy Efficiency [Mbit/Joule]');


%Plot Figure 12: Total RF power at the EE-maximizing solution in the
%multi-cell scenario for different number of antennas. The radiated power per
%BS antennas is also shown.
figure(12); hold on; box on;
title('Figure 12: Multi-cell, Comparison of RF power and radiated power/antenna')

plot(Mrange,optRFpowersMulticellReuse4,'b--','LineWidth',1);
plot(Mrange,optRFpowersMulticellReuse2,'k','LineWidth',1);
plot(Mrange,optRFpowersMulticellReuse1,'r-.','LineWidth',1);

plot(MoptimalReuse4,optRFpowersMulticellReuse4(MoptimalReuse4),'bo','LineWidth',1);
plot(MoptimalReuse2,optRFpowersMulticellReuse2(MoptimalReuse2),'ko','LineWidth',1);
plot(MoptimalReuse1,optRFpowersMulticellReuse1(MoptimalReuse1),'ro','LineWidth',1);


plot(Mrange,zetaDL*eta*optRFpowersMulticellReuse1./Mrange,'r-.','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersMulticellReuse2./Mrange,'k','LineWidth',1);
plot(Mrange,zetaDL*eta*optRFpowersMulticellReuse4./Mrange,'b--','LineWidth',1);

plot(MoptimalReuse4,zetaDL*eta*optRFpowersMulticellReuse4(MoptimalReuse4)./MoptimalReuse4,'bo','LineWidth',1);
plot(MoptimalReuse2,zetaDL*eta*optRFpowersMulticellReuse2(MoptimalReuse2)./MoptimalReuse2,'ko','LineWidth',1);
plot(MoptimalReuse1,zetaDL*eta*optRFpowersMulticellReuse1(MoptimalReuse1)./MoptimalReuse1,'ro','LineWidth',1);

set(gca,'YScale','Log');
axis([0 Mmax 1e-2 1e2]);

text(20,5.5,'Total RF power');
text(20,0.15,'Radiated power per BS antenna');
legend('ZF (Imperfect CSI): Reuse 4','ZF (Imperfect CSI): Reuse 2','ZF (Imperfect CSI): Reuse 1','Location','Best');

xlabel('Number of Antennas (M)');
ylabel('Average Power [W]');


%Plot Figure 13: Area throughput at the EE-maximizing solution in the
%multi-cell scenario, for different number of BS antennas.
figure(13); hold on; box on;
title('Figure 13: Multi-cell, Comparison of area throughput')

plot(Mrange,optEEsumratesMulticellReuse4/areaMulticell,'b--','LineWidth',1);
plot(Mrange,optEEsumratesMulticellReuse2/areaMulticell,'k','LineWidth',1);
plot(Mrange,optEEsumratesMulticellReuse1/areaMulticell,'r-.','LineWidth',1);

plot(MoptimalReuse4,optEEsumratesMulticellReuse4(MoptimalReuse4)/areaMulticell,'bo','LineWidth',1);
plot(MoptimalReuse2,optEEsumratesMulticellReuse2(MoptimalReuse2)/areaMulticell,'ko','LineWidth',1);
plot(MoptimalReuse1,optEEsumratesMulticellReuse1(MoptimalReuse1)/areaMulticell,'ro','LineWidth',1);

axis([0 Mmax 0 9]);

legend('ZF (Imperfect CSI): Reuse 4','ZF (Imperfect CSI): Reuse 2','ZF (Imperfect CSI): Reuse 1','Location','NorthWest')

xlabel('Number of Antennas (M) ');
ylabel('Area Throughput [Gbit/s/km^2]');



%Plot Figure 14: Energy efficiency (in Mbit/Joule) with ZF processing
%in multi-cell scenario with pilot reuse 4.
figure(14); grid on; hold on;
title('Figure 14: ZF processing, Multi-cell, Pilot reuse 4')

surface(1:Kmax,1:Mmax,EEoptZFMulticellReuse4(1:Mmax,:)/1e6,'EdgeColor','none'); %Plot the 3d surface
colormap(autumn);

%Compute and plot the optimal point
[EEvalues,indM] = max(EEoptZFMulticellReuse4,[],2);
[EEoptimal,indK] = max(EEvalues);
plot3(indM(indK),indK,EEoptimal/1e6,'k*','MarkerSize',10);

%Plot lines on top of the 3d surface, to make it easier to see the shape
for m = [1 gridDensity:gridDensity:Mmax]
    plot3(1:Kmax,m*ones(1,Kmax),EEoptZFMulticellReuse4(m,:)/1e6,'k-');
end

for k = [1 gridDensity:gridDensity:Kmax]
    plot3(k*ones(1,Mmax),1:Mmax,EEoptZFMulticellReuse4(:,k)/1e6,'k-');
end

plot3(1:Kmax,1:Kmax,zeros(Kmax,1),'k-');

view([-46 24])
axis([0 Kmax 0 Mmax 0 8])

ylabel('Number of Antennas (M)');
xlabel('Number of Users (K)');
zlabel('Energy Efficiency [Mbit/Joule]');
