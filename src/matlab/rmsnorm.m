function signal = rmsnorm(input_signal)
    %quick script to rms normalize wav signals
    % MJM, November 15th, 2017 
    c = input_signal;
    desired_rms = .05;
    rmsx = norm(c)/sqrt(length(c));
    signal = c/rmsx*desired_rms;

end
