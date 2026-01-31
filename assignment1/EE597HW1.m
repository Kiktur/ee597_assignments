%% EE 597 HW 1


%% Problem 1
% Using the simple path loss model, and assuming SNR = Eb/No (i.e. the spectral efficiency R/W =
% 1), plot the Bit Error Rate of BPSK (in log-scale) as a function of distance if PT = 0 dBm, Kref, dB =
% -20dB, d0 = 1m, for η = 2 and η = 4; assume that noise power = -80dBm. Comment on the plot.
% Include distances sufficiently large to let the BER rise to close to 0.5.

clc; clear; close all;

pt_db = 0;
kref_db = -20; 
d0 = 1;

d = 1:1:7000; % 1 to 7000 meters

eta_vals = [2, 4];
N0 = -80;

figure;
hold on;


for eta = eta_vals
    % Simple path loss model
    pr_db = pt_db + kref_db - (eta * 10 * (log10(d/d0)));
    
    % SNR in db
    ebno = pr_db - N0; 
    
    % Convert to linear
    ebno_lin = 10 .^ (ebno / 10);

    % BPSK bit error rate
    BER = qfunc(sqrt(2 * ebno_lin)); 

    % Logarithmic plot
    semilogy(d, BER, 'LineWidth', 2);
end

xlabel('Distance (m)');
ylabel('Bit Error Rate (BER)');

title('BER vs Distance'); 

legend('\eta = 2', '\eta = 4', 'Location', 'southwest');
ylim([1e-5 1]);



%% Problem 2

% Now consider a wireless channel with log-normal fading with a given standard deviation 𝜎dB .
% Numerically calculate and plot the expected BER as a function of distance (set η = 2, using all
% the other parameters the same as #2 above) as 3 separate curves for 𝜎dB = 5dB, 10dB and 20dB,
% and on the same plot include your plot from #2 above. How do the curves compare? Comment on
% the plot.
% Hint: you can do this plot without doing simulations. Use the integral definition of E[BER]
% (which is the integral of the product of the BER curve and the fading distribution), and use
% a numerical method to calculate the integral.


sigma_vals = [5, 10, 20];

d = 1:1:7000; % 1 to 7000 meters

N0 = -80; % Noise power

pt_db   = 0;    
kref_db = -20;    
d0       = 1;     
eta      = 2;      

pr_db = pt_db + kref_db - 10*eta*log10(d/d0);

ebno_db = pr_db - N0;

ebno_lin = 10.^(ebno_db/10);

BER_no_fading = qfunc(sqrt(2 * ebno_lin));

figure; hold on;

semilogy(d, BER_no_fading, 'k', 'LineWidth', 2);

for k = 1:length(sigma_vals)
    sigma = sigma_vals(k);
    
    for i = 1:length(d)
        % ebno at distance d
        gamma_bar_db = ebno_db(i);
        
        % Includes f(gamma) and g(gamma)
        integrand = @(x) qfunc(sqrt(2 * 10.^((gamma_bar_db + x)/10))) .* (1/(sqrt(2*pi)*sigma)) .* exp(-x.^2/(2*sigma^2));
        
        BER_avg(k,i) = integral(integrand, -5*sigma, 5*sigma);
    end
    
    semilogy(d, BER_avg(k,:), 'LineWidth', 2);
end

grid on;
xlabel('Distance (m)');
ylabel('Bit Error Rate (BER)');
title('Average BER vs Distance with Log-Normal Fading');
legend('No fading', '\sigma = 5 dB', '\sigma = 10 dB', '\sigma = 20 dB', ...
       'Location', 'southwest');
ylim([1e-5 1]);


%% Problem 3

% Using the Shannon capacity formula, plot the maximum data rate as a function of distance for η =
% 2 and all other radio parameters the same as question # 2 above (assuming the simple path loss
% model, no fading) if the channel bandwidth is 20 MHz. Comment on the plot. Additionally, try to
% come up with a simple but approximately correct equation to describe how the rate varies
% with distance and show your approximate curve also on the same plot. Note that this does
% not have one right answer, many approximate equations may exist, but try to make your
% equation as simple and general as possible (would be ideal if your equation can explicitly
% account for the parameters PT , η).

% clc; clear; close all;

pt_db   = 0;    
kref_db = -20;     
d0       = 1;   
eta      = 2;    
N0   = -80;       

d = 1:1:7000; % 1 to 7000 meters

pr_db = pt_db + kref_db - 10*eta*log10(d/d0);
ebno_db = pr_db - N0;
ebno_lin = 10.^(ebno_db/10);


W = 20 * 10^6;

% SNR = ebno in LINEAR scale
% Shannon capacity theorem
C = W * log2(1 + ebno_lin);
% C is in bits per second

figure; 
hold on;
plot(d, C, 'LineWidth', 2);
grid on;
xlabel('Distance (m)');
ylabel('Data Rate (bits/sec)');
title('Maximum Data Rate vs Distance');


% TODO: Try to make equation to describe relationship




%% Problem 4
% 
% Again consider log-normal fading with 𝜎dB. Assuming that an SNR below 10dB results in an
% outage event (i.e. is unacceptable), plot the outage probability (not in log-scale) as a function of
% distance as three separate curves for 𝜎dB = 5dB, 10dB and 20dB (again, assuming η = 2 and
% using all the other parameters the same as #2 above).


eta = 2;

sigma_vals = [5, 10, 20];

pt_db   = 0; 
kref_db = -20;   
d0       = 1;   
N0   = -80;      

d = 1:1:7000; % 1 to 7000 meters

pr_db = pt_db + kref_db - 10*eta*log10(d/d0);
ebno_db = pr_db - N0;
ebno_lin = 10.^(ebno_db/10);

threshold_db = 10; 

figure; hold on;

% Pout = 1 - Q( (threshold_db - (expected_val(Pr_db(d)) - NdBm)) /
% sigma_db)

for sigma = sigma_vals
    qfunc_exp = (threshold_db - ebno_db) / sigma;
    Pout = 1 - qfunc(qfunc_exp);
    plot(d, Pout, 'LineWidth', 2);

end

% Plot formatting
grid on;
xlabel('Distance (m)');
ylabel('Probability of Outage');
title('Probability of Outage vs Distance with Threshold = 10 dB');
legend('\sigma = 5 dB', '\sigma = 10 dB', '\sigma = 20 dB', ...
       'Location', 'southwest');


% TODO: Double check if these plots seem right





%% Problem 5

% The provided data set is taken from experiments conducted in the fourth floor of RTH. It contains
% received signal strengths from 802.11 devices located at various fixed locations. Assume that the
% signal strength decays according to the simplified path-loss model with log-normal fading. Using
% the data set, estimate the parameters for the path loss at reference distance (assume d0 =
% 1m) Kref,dB, the path loss exponent η , and the fading standard deviation 𝜎dB. The data set is
% described below and posted on the Google drive along with this assignment:
