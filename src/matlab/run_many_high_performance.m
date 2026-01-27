function run_many_high_performance(input_path)
% RUN_MANY_HIGH_PERFORMANCE Process multiple audio files with prosody analysis
% Usage: run_many_high_performance('data/ja.txt')
%        run_many_high_performance('/path/to/data_folder')
%
% If a directory is provided, finds all .txt files and processes them.
% If a file is provided, reads audio paths from that file.
% Outputs are saved to the directory specified in config.json.

STRAIGHT_folder = 'prosody_meaning/STRAIGHT/TandemSTRAIGHTmonolithicPackage010';

% Add STRAIGHT path once at the beginning
addpath(genpath(STRAIGHT_folder));

% Validate input argument
if nargin < 1
	error('Usage: run_many_high_performance(input_path)');
end

% Basic environment validation
if ~exist(STRAIGHT_folder, 'dir')
	error('STRAIGHT folder not found: %s', STRAIGHT_folder);
end
if exist('synthOnlyProsody', 'file') ~= 2
	warning('Function synthOnlyProsody not found on path. Ensure it is accessible.');
end

% Load config (language -> output directory mapping)
fid = fopen('config.json');
raw = fread(fid, inf);
str = char(raw');
fclose(fid);
cfg = jsondecode(str);

% Collect all audio file paths
file_paths_all = {};
output_paths_all = {};

if isfolder(input_path)
	% Directory mode: find all .txt files and read paths from each
	fprintf('Scanning directory: %s\n', input_path);
	txt_lists = dir(fullfile(input_path, '**', '*.txt'));
	
	if isempty(txt_lists)
		error('No .txt files found under directory: %s', input_path);
	end
	
	for k = 1:numel(txt_lists)
		list_path = fullfile(txt_lists(k).folder, txt_lists(k).name);
		
		% Extract language from filename (e.g., 'ja.txt' -> 'ja')
		[~, lang, ~] = fileparts(txt_lists(k).name);
		output_root = get_output_dir(cfg, lang);
		
		fprintf('Reading: %s (language: %s)\n', list_path, lang);
		[paths, outputs] = read_audio_list(list_path, output_root);
		file_paths_all = [file_paths_all; paths]; %#ok<AGROW>
		output_paths_all = [output_paths_all; outputs]; %#ok<AGROW>
	end
else
	% Single file mode: read paths from the provided txt file
	if ~exist(input_path, 'file')
		error('File ''%s'' does not exist.', input_path);
	end
	
	% Extract language from filename (e.g., 'ja.txt' -> 'ja')
	[~, lang, ~] = fileparts(input_path);
	output_root = get_output_dir(cfg, lang);
	
	fprintf('Reading: %s (language: %s)\n', input_path, lang);
	[file_paths_all, output_paths_all] = read_audio_list(input_path, output_root);
end

fprintf('Total audio files to process: %d\n', numel(file_paths_all));

% Configuration for high-performance server
total_files = numel(file_paths_all);
if total_files == 0
	fprintf('No files to process.\n');
	return;
end

available_cores = feature('numcores');
max_cores = min(100, max(1, available_cores));
batch_size = max_cores;
num_batches = ceil(total_files / batch_size);

fprintf('=== HIGH-PERFORMANCE PROCESSING ===\n');
fprintf('Total files: %d\n', total_files);
fprintf('Cores allocated: %d\n', max_cores);
fprintf('Batch size: %d\n', batch_size);
fprintf('Number of batches: %d\n', num_batches);
fprintf('Estimated total time: %.1f minutes\n', (total_files * 0.5) / max_cores / 60);

% Start parallel pool (use process-based pool, not threads, because STRAIGHT modifies MATLAB path)
pool = gcp('nocreate');
if isempty(pool)
	if strcmp(getenv('NO_PARPOOL'), '1')
		fprintf('NO_PARPOOL=1 set; running without parallel pool.\n');
	else
		try
			pool = parpool('local', max_cores);
			fprintf('Process-based parallel pool started with %d workers\n', pool.NumWorkers);
		catch ME
			fprintf('Failed to start parallel pool: %s\n', ME.message);
			fprintf('Proceeding without a parallel pool; parfor will execute serially.\n');
		end
	end
end

total_start_time = tic;
total_successful = 0;
total_failed = 0;

for batch = 1:num_batches
	batch_start_idx = (batch - 1) * batch_size + 1;
	batch_end_idx = min(batch * batch_size, total_files);
	batch_files = batch_end_idx - batch_start_idx + 1;
	
	fprintf('\n=== BATCH %d/%d: Processing files %d-%d (%d files) ===\n', ...
		batch, num_batches, batch_start_idx, batch_end_idx, batch_files);
	
	% Pre-allocate for this batch
	file_paths = cell(batch_files, 1);
	batch_output_paths = cell(batch_files, 1);
	success_flags = false(batch_files, 1);
	error_messages = cell(batch_files, 1);
	
	% Extract file paths for this batch
	for i = 1:batch_files
		file_paths{i} = file_paths_all{batch_start_idx + i - 1};
		batch_output_paths{i} = output_paths_all{batch_start_idx + i - 1};
	end
	
	% Process batch in parallel
	batch_start_time = tic;
	parfor i = 1:batch_files
		try
			output_path = batch_output_paths{i};
			
			% Check if output already exists
			if exist(output_path, 'file')
				success_flags(i) = true;
				continue;
			end
			
			% Validate input file exists
			if ~exist(file_paths{i}, 'file')
				error('Input audio not found: %s', file_paths{i});
			end
			
			% Ensure output directory exists
			output_dir = fileparts(output_path);
			if ~exist(output_dir, 'dir')
				mkdir(output_dir);
			end

			% Process the file and write directly to output_path
			synthOnlyProsody(file_paths{i}, STRAIGHT_folder, output_path);

			% Verify output exists
			if ~exist(output_path, 'file')
				error('Output not created: %s', output_path);
			end

			success_flags(i) = true;

			
		catch ME
			error_messages{i} = ME.message;
		end
	end
	
	batch_time = toc(batch_start_time);
	batch_successful = sum(success_flags);
	batch_failed = batch_files - batch_successful;
	
	total_successful = total_successful + batch_successful;
	total_failed = total_failed + batch_failed;
	
	% Calculate progress and ETA
	progress = min((batch * batch_size) / max(total_files, 1) * 100, 100);
	elapsed_total = toc(total_start_time);
	eta_seconds = (elapsed_total / batch) * (num_batches - batch);
	eta_minutes = eta_seconds / 60;
	
	fprintf('Batch %d completed in %.2f seconds\n', batch, batch_time);
	fprintf('  Successful: %d, Failed: %d\n', batch_successful, batch_failed);
	fprintf('  Progress: %.1f%% (ETA: %.1f minutes)\n', progress, eta_minutes);
	
	% Print sample error messages for failed items
	if batch_failed > 0
		fprintf('  Errors in this batch (up to 10 shown):\n');
		shown = 0;
		for i = 1:batch_files
			if ~success_flags(i) && ~isempty(error_messages{i})
				fprintf('    - %s\n', error_messages{i});
				shown = shown + 1;
				if shown >= 10
					break;
				end
			end
		end
	end
	
	% Clear variables to free memory
	clear file_paths batch_output_paths success_flags error_messages;
end

total_time = toc(total_start_time);

% Final summary
fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('Total files processed: %d\n', total_files);
fprintf('Successful: %d\n', total_successful);
fprintf('Failed: %d\n', total_failed);
fprintf('Total time: %.2f seconds (%.2f minutes)\n', total_time, total_time/60);
if total_files > 0
	fprintf('Average time per file: %.2f seconds\n', total_time/total_files);
else
	fprintf('Average time per file: N/A (no files)\n');
end
fprintf('Effective speedup: %.1fx\n', (total_files * 10) / max(total_time, 1));
fprintf('Cores used: %d\n', max_cores);

end


function output_dir = get_output_dir(cfg, lang)
% GET_OUTPUT_DIR Look up output directory for a language from config
%   Returns the directory path, creating it if needed

if isfield(cfg, lang)
	output_dir = cfg.(lang);
else
	error('Language ''%s'' not found in config.json. Available: %s', ...
		lang, strjoin(fieldnames(cfg), ', '));
end

% Create output directory if needed
if ~exist(output_dir, 'dir')
	mkdir(output_dir);
	fprintf('Created output directory: %s\n', output_dir);
end

end


function [file_paths, output_paths] = read_audio_list(list_path, output_root)
% READ_AUDIO_LIST Read audio file paths from a text file
%   Returns cell arrays of input paths and corresponding output paths

file_paths = {};
output_paths = {};

data = readtable(list_path, 'Delimiter', '\n', 'ReadVariableNames', false);
[list_dir, ~, ~] = fileparts(list_path);

for r = 1:height(data)
	this_audio = strtrim(data{r, 1}{1});
	
	% Skip empty or commented lines
	if isempty(this_audio) || startsWith(this_audio, '#')
		continue;
	end
	
	% Resolve relative paths against the list file's directory
	if ~startsWith(this_audio, filesep)
		this_audio = fullfile(list_dir, this_audio);
	end
	
	% Generate output path
	[~, base_name, ~] = fileparts(this_audio);
	output_filename = [base_name '.wav'];
	
	file_paths{end+1, 1} = this_audio; %#ok<AGROW>
	output_paths{end+1, 1} = fullfile(output_root, output_filename); %#ok<AGROW>
end

end
