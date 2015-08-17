function EEvalue = functionCalculateEmpircalEE(SINR,Ddiag,Doffdiag,circuitpower,prelogFactor,A,prefixPower)
%Compute the highest achievable EE by optimizing the power allocation to
%achieve a given SINR per UE (or equally a given gross rate) for a given
%processing scheme and system dimensions. Note that it can only be applied
%for processing schemes that are independent of the power allocation (which
%is not the case for MMSE processing). The power allocation is computed as
%described in Eq. (8) and Eq. (16).
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
%SINR         = Scalar SINR value that all UEs should achieve (equal to 2^(Rbar/B)-1)
%Ddiag        = Diagonal elements of D^(ul) for each Monte Carlo
%               realization (except for the factor 1/(2^(Rbar/B)-1)) for
%               an arbitrary processing scheme
%Doffdiag     = Off-diagonal elements of D^(ul) for each Monte Carlo
%               realization for an arbitrary processing scheme
%circuitpower = Circuit power (i.e., excluding RF power) with a certain
%               processing scheme and number of antennas/UEs
%prelogFactor = Factor in front of the logarithm in the rate expression in Eq. (32)
%A            = Parameter A in the power consumption model (see Table 1)
%prefixPower  = Factor in front of the total RF power in Eq. (19)
%
%OUTPUT:
%EEvalue      = Energy efficiency achieved for given processing scheme and SINRs


nbrOfMonteCarloRealizations = size(Ddiag,3); %Extract number of Monte Carlo realizations
oneK = ones(size(Ddiag,1),1); %Create the vector 1_K with K ones.

%Placeholder for the sum transmit power in the uplink/downlink
sumPower = zeros(nbrOfMonteCarloRealizations,1);

%Compute the matrix in D^(ul) in Eq. (9) for a certain SINR per UE.
D = Ddiag/SINR + Doffdiag;

%Go through all Monte Carlo realizations
for itr = 1:nbrOfMonteCarloRealizations
    
    %Compute uplink user powers according to Eq. (8).
    userPowers = D(:,:,itr)\oneK;
    
    %Check if the SINR is feasible/achievable for all UEs, which is the
    %case if and only if the user powers are all positive.
    if min(userPowers)>0
        
        %Compute the sum of the uplink user powers. Note that the sum of
        %the downlink powers are the same, due to uplink-downlink duality.
        sumPower(itr) = sum(userPowers); 
        
    else
        
        %Assign a negative value when the SINRs are infeasible.
        sumPower(itr) = -1;
        
        %Stop the process, since one infeasible scenario is sufficient for
        %infeasibility.
        break;
        
    end
    
end


%Check if all the SINR was feasible for Monte Carlo realizations
if min(sumPower)>0
    
    %Compute the average transmit power
    averagePower = mean(sumPower);
    
    %Compute the corresponding EE
    EEvalue = ( prelogFactor*log2(1+SINR) ) / (  prefixPower*averagePower + circuitpower + A*prelogFactor*log2(1+SINR) ); 
    
else
    
    %Set the EE to zero when the current SINRs are infeasible
    EEvalue = 0;
    
end
