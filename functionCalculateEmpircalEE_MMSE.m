function EEvalue = functionCalculateEmpircalEE_MMSE(SINR,Hchannel,circuitpower,A,prelogFactor,prefixPower,sigma2B,l_x,Q)
%Compute the highest achievable EE by optimizing the power allocation to
%achieve a given SINR per UE (or equally a given gross rate) for MMSE-like
%processing and some given system dimensions. The power allocation is
%updated Q-1 times in the process, which might not be sufficient to find
%the true MMSE processing filter but sufficiently close since the power
%control is determined by fixed-point iterations using interference functions.
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
%Hchannel     = Channel realizations for the Monte Carlo realizations 
%circuitpower = Circuit power (i.e., excluding RF power) with MMSE processing
%               and fixed number of antennas/UEs
%A            = Parameter A in the power consumption model (see Table 1)
%prelogFactor = Factor in front of the logarithm in the rate expression in Eq. (32)
%prefixPower  = Factor in front of the total RF power in Eq. (19)
%sigma2B      = Total noise power (B*sigma2 in W)
%l_x          = Average pathlosses for the UEs in the Monte Carlo realizations
%Q            = Number of iterations in power allocation computation
%
%OUTPUT:
%EEvalue      = Energy efficiency achieved for MMSE processing and given SINRs


M = size(Hchannel,1); %Extract number of BS antennas
K = size(Hchannel,2); %Extract number of UEs

nbrOfMonteCarloRealizations = size(Hchannel,3); %Extract number of Monte Carlo realizations
oneK = ones(K,1); %Create the vector 1_K with K ones.

%Placeholder for the sum transmit power in the uplink/downlink
sumPower = zeros(nbrOfMonteCarloRealizations,1);

%Go through all Monte Carlo realizations
for itr = 1:nbrOfMonteCarloRealizations
    
    %Initiate the power allocation
    if M>K
        powerAllocation = (SINR/(M-K))./l_x(:,itr); %This is a good starting point when ZF exists.
    else
        powerAllocation = SINR./l_x(:,itr); %This is an alternative starting point when ZF doesn't exist.
    end
    
    %Update the power allocation Q-1 times by exploiting that Eq. (8) is a
    %fixed point iteration. More exactly, we use the conventional power
    %control algorithm from the following article:
    %
    %Farrokh Rashid-Farrokhi, K.J. Ray Liu, and Leandros Tassiulas,
    %“Transmit beamforming and power control for cellular wireless systems,”
    %IEEE J. Select. Areas Commun., vol. 16, no. 8, pp. 1437-1450, Oct. 1998.
    for m = 1:Q-1
        
        %Placeholder for the next power allocation
        nextPowerAllocation = zeros(K,1);
        
        %Precompute the covariance matrix of the received signal with
        %current power allocation
        S = (Hchannel(:,:,itr)*diag(powerAllocation)*Hchannel(:,:,itr)'+eye(M));
        
        %Go through each of the users
        for k = 1:K
            
            %Compute the MMSE filter of UE k with current power allocation
            vector = (S-powerAllocation(k)*Hchannel(:,k,itr)*Hchannel(:,k,itr)')\Hchannel(:,k,itr);
            
            %Compute which SINR that UE k achieves
            SINRk = powerAllocation(k)*(Hchannel(:,k,itr)'*vector);
            
            %Update the power of UE k based on the ratio between the achieved
            %SINRk and the SINR that the user should achieve
            nextPowerAllocation(k) = powerAllocation(k)*SINR/SINRk;
            
        end
        
        %Update power allocation
        powerAllocation = nextPowerAllocation;
    end
    
    %Compute the matrix D^(ul) in Eq. (9) for MMSE processing
    G_MMSE = (Hchannel(:,:,itr)*diag(powerAllocation)*Hchannel(:,:,itr)'+eye(M))\Hchannel(:,:,itr); %Compute MMSE matrix
    G_MMSE = G_MMSE ./ repmat(sqrt(sum(abs(G_MMSE).^2,1)),[M 1]); %Normalize each column
    
    %Multiply the MMSE matrix with all the current channels and normalize by noise power
    Gains = abs(Hchannel(:,:,itr)'*G_MMSE).^2/sigma2B;
    
    DdiagMMSE = diag(diag(Gains))/SINR; %Compute the diagonal elements of D^(ul) for current Monte Carlo realization
    DoffdiagMMSE = diag(diag(Gains)) - Gains; %Compute the off-diagonal elements of D^(ul) for current Monte Carlo realization
    D = DdiagMMSE + DoffdiagMMSE; %Put together the matrix D^(ul)
    
    %Compute uplink user powers according to Eq. (8).
    userPowers = D\oneK;
    
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
    EEvalue = ( prelogFactor*log2(1+SINR) ) / (  prefixPower*averagePower + circuitpower + A*prelogFactor*log2(1+SINR) ); %Scale to get a reasonable scale!
    
else
    
    %Set the EE to zero when the current SINRs are infeasible
    EEvalue = 0;
    
end
