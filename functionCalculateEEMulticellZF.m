function [EEvalue,sumRate,averageRFPower] = functionCalculateEEMulticellZF(alpha,M,K,Ijl_PC,Ijl_nonPC,tauUL,Bsigma2Sxeta,A,circuitpower,prelogFactor)
%Compute the achievable rate in single-cell and multi-cell scenarios with
%ZF processing and imperfect CSI. This is an implementation of Eq. (58) in
%Lemma 6, and Eq. (55) in Lemma 5 is achieved as a special case. This
%function can be used to optimize the EE by varying different parameters.
%
%This function is used in the article:
%
%Emil Björnson, Luca Sanguinetti, Jakob Hoydis, Mérouane Debbah, “Optimal
%Design of Energy-Efficient Multi-User MIMO Systems: Is Massive MIMO the
%Answer?,” IEEE Transactions on Wireless Communications, vol. 14, no. 6, 
%pp. 3059-3075, June 2015. 
%
%Download article:
%
%This is version 1.0 (Last edited: 2014-03-21)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above.
%
%
%INPUT:
%alpha        = Design parameter that is proportional the RF power
%M            = Number of BS antennas
%K            = Number of UEs
%Ijl_PC       = Vector with average relative channel attenuations from the
%               interfering cells that use the same pilot sequences. The
%               single-cell case is achieved by setting Ijl_PC = []
%Ijl_nonPC    = Vector with average relative channel attenuations from the
%               interfering cells that use orthogonal pilot sequences. The
%               single-cell case is achieved by setting Ijl_nonPC = []
%tauUL        = Relative pilot length in the uplink (1, 2, 4 are typical
%               values that correspond to different pilot reuse patterns 
%Bsigma2Sxeta = The term B*sigma^2*S_x/eta that appears in Eq. (19) and
%               other places that defines the total RF power
%A            = Parameter A in the power consumption model (see Table 1)
%circuitpower = Circuit power (i.e., excluding RF power) for ZF processing
%               with the given numbers of antennas and UEs
%prelogFactor = Factor in front of the logarithm in the rate expression in Eq. (32)
%
%OUTPUT:
%EE             = Energy efficiency achieved for given processing scheme and SINRs
%sumRate        = Sum rate that is achieved for the given configuration
%averageRFPower = Average RF power that is used to achieve the EE


%The design parameter alpha (proportional to transmit power) must be positive
if alpha >= 0
    
    %Vector with average relative channel attenuations from the interfering
    %cells that cause pilot contamination and for the serving cell itself,
    %which gives a relative attenuation of one.
    Ijl_PC_Extended = [Ijl_PC; 1];
    
    %Compute the SINR that each UE achieves, according to Lemma 6.
    SINR = alpha*(M-K) / ( alpha*(M-K)*sum(Ijl_PC) + (1+sum(Ijl_PC)+1/(alpha*K*tauUL)) * ( 1 + K*alpha*sum(Ijl_nonPC) + K*alpha*( sum(Ijl_PC_Extended)^2 - sum(Ijl_PC_Extended.*Ijl_PC_Extended) + 1/(alpha*K*tauUL)   )/( sum(Ijl_PC_Extended) + 1/(alpha*K*tauUL) )  ) );
    
    sumRate = prelogFactor*log2(1+SINR); %Compute the corresponding sum rate
    averageRFPower = Bsigma2Sxeta*K*alpha; %Compute the corresponding RF power
    EEvalue = sumRate / ( averageRFPower + circuitpower + A*sumRate ); %Compute the corresponding EE
    
else
    
    %All the parameters are zero when alpha<0, which means that there is no
    %transmit power and thus the EE and rates are zero.
    EEvalue = 0;
    sumRate = 0;
    averageRFPower = 0;
    
end
