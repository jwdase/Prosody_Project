function synthOnlyProsody(inputsig, STRAIGHTfolder, outputsig)
%synth complex sound with envelope and f0 extracted from a speech signal
% Tamar Regev July 13 2023
%
%  Requirements:
%   STRAIGHT can be downloaded from: https://github.com/tamaregev/prosody_meaning/tree/master/STRAIGHT
%   generate_singlenote_vary_envelope_jitter_randphase_Tamar.m
%   rmsnorm.m
%   hann.m

%  INPUT:
%   inputsig = 'PATH/common_voice_en_3879.wav';
%   STRAIGHTfolder = path to STRAIGHT;

%  OUTPUT:
%   The output sound file is saved in the same folder as
%   inputsig, added '_prosody' to its name (assuming .wav extension)
    %% definitions
    addpath(genpath(STRAIGHTfolder))
    %% analyze voice
    [y,fs]=audioread(inputsig);
    
    % STRAIGHT analysis:
    % STRAIGHT analysis continued:
    % These are the features we extract from audio file

    r = exF0candidatesTSTRAIGHTGB(y,fs); % Extract F0 information
    rc = autoF0Tracking(r,y); % Clean F0 trajectory by tracking


    rc.vuv = refineVoicingDecision(y,rc); % Takes in y, rc and 
    q = aperiodicityRatioSigmoid(y,rc,1,2,0); % aperiodicity extraction
    
    r_speech = r;
    rc_speech = rc;
    q_speech = q;
    
    %% extract speech amplitude envelope
    
    % loudness trajectory
    
    wav_Pa = y * 1;
    smooth_sec = 0.125;  %"FAST" SPL is 1/8th of second.  "SLOW" is 1 second;
    smooth_Hz = 1/smooth_sec;
    
    [b,a]=butter(1,smooth_Hz/(fs/2),'low');  %design a Low-pass filter
    wav_env_Pa = sqrt(filter(b,a,wav_Pa.^2));  %rectify, by squaring, and low-pass filter
    
    %compute SPL
    %Pa_ref = 20e-6;  %reference pressure for SPL in Air
    %SPL_dB = 10.0*log10( (wav_env_Pa ./ Pa_ref).^2 ); % 10*log10 because signal is squared
    
    %% generate complex tone
    
    len = length(y)/fs;%sec
    
    f0 = 200;
    harm_nums = 1:100;
    %harm_nums = 1;
    %jitt_amt =.5;
    jitt_amt = 0;
    jitt = 0; % 2 to generate a single note with jitter (1 requires JitterString input)
    dur_s = len;
    sr = fs;
    dist = 0;
    JitterString = [];
    centroid = [];
    
    [signal, ~] = generate_singlenote_vary_envelope_jitter_randphase_Tamar(f0, harm_nums, jitt_amt,jitt, dur_s, sr, dist,JitterString, centroid);
    
    %% analyze comp tone
    signal = signal';
    r = exF0candidatesTSTRAIGHTGB(signal,fs); % Extract F0 information
    rc = autoF0Tracking(r,signal); % Clean F0 trajectory by tracking
    rc.vuv = refineVoicingDecision(signal,rc);
    q = aperiodicityRatioSigmoid(signal,rc,1,2,0); % aperiodicity extraction
    f = exSpectrumTSTRAIGHTGB(signal,fs,q);
    
    q.f0 = rc_speech.f0;
    
    %% and resynth with f0 and amp of voice
    % 
    s2 = exGeneralSTRAIGHTsynthesisR2(q,f); % new implementation
    
    % impse the f0 trajectory on the complex tone:
    sound_out = s2.synthesisOut/max(abs(s2.synthesisOut))*0.8;
    
    %%%%%%%%%%%%%%%%%%% multiply sound_out by the amplitude:
    
    %this one sounds good:
    sound_out_loud = sound_out.* wav_env_Pa(1:length(sound_out));
    sound_out_loud = sound_out_loud.*5;

    % Move the location 

    audiowrite([outputsig],sound_out_loud,fs)

end
