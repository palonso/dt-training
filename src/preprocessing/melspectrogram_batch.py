import os
import sys
from argparse import ArgumentParser
from multiprocessing import Pool
from multiprocessing import cpu_count
from subprocess import run, PIPE
from essentia import EssentiaError
from functools import partial
import os
import sys
from tqdm import tqdm


def _subprocess(cmd, verbose=True):
    """General purpose subprocess."""

    completed_process = run(cmd, stdout=None, stderr=PIPE)

    cmd_str = ' '.join([str(x) for x in cmd])
    stderr = completed_process.stderr.decode("utf-8")
    rc = completed_process.returncode

    if verbose:
        if rc == 0:
            print('"{}"... ok!'.format(cmd_str))
        else:
            print('"{}"... failed (returncode {})!'.format(cmd_str, rc))
            print(stderr, '\n')

    return rc, cmd_str, stderr

def _batch_extractor(audio_dir, output_dir, extractor_cmd, output_extension,
                     generate_log=True, audio_types=None, skip_analyzed=False,
                     jobs=0, verbose=True, blacklist_file=None,
                     whitelist_file=None, target=1000000):
    blacklist, whitelist = None, None
    if not audio_types:
        audio_types = ('.wav', '.aiff', '.flac', '.mp3', '.ogg')
        print("Audio files extensions considered by default: " +
              ', '.join(audio_types))
    else:
        if type(audio_types) == str:
            audio_types = [audio_types]

        audio_types = tuple(audio_types)
        audio_types = tuple([i.lower() for i in audio_types])
        print("Searching for audio files extensions: " + ', '.join(audio_types))
    print("")

    output_dir = os.path.abspath(output_dir)
    audio_dir = os.path.abspath(audio_dir)

    if blacklist_file:
        with open(blacklist_file, 'r') as f:
            blacklist = set([l.rstrip() for l in f.readlines()])
        print(f'there are ({len(blacklist)}) existing files')

    if whitelist_file:
        with open(whitelist_file, 'r') as f:
            whitelist = set([l.rstrip() for l in f.readlines()])
        print(f'computing spectrograms for ({len(whitelist)}) files')

    if jobs == 0:
        try:
            jobs = cpu_count()
        except NotImplementedError:
            print('Failed to automatically detect the cpu count, '
                  'the analysis will try to continue with 4 jobs. '
                  'For a different behavior change the `job` parameter.')
            jobs = 4

    skipped_count = 0
    skipped_files = []
    cmd_lines = []

    already_computed = set()
    ids_to_compute = list()
    pbar = tqdm()
    for abs_file_dir, _, filenames in os.walk(audio_dir):
        for filename in filenames:
            if filename.lower().endswith(audio_types):
                rel_file_dir = os.path.relpath(abs_file_dir, start=audio_dir)
                audio_file = os.path.join(abs_file_dir, filename)
                out_file = os.path.join(output_dir, rel_file_dir, filename)

                if blacklist:
                    if filename in blacklist:
                        # we can remove the key from blacklist to speed up
                        blacklist.discard(filename)
                        pbar.update()
                        continue

                if whitelist:
                    if filename not in whitelist:
                        print(f'{filename} not in the whitelist')
                        pbar.update()
                        continue

                if skip_analyzed:
                    if os.path.isfile( '{}.{}'.format(out_file, output_extension)):
                        print("Found descriptor file for " +
                              audio_file + ", skipping...")
                        skipped_files.append(audio_file)
                        skipped_count += 1
                        already_computed.add(filename)
                        target -= 1
                        pbar.update()
                        continue

                folder = os.path.dirname(out_file)

                if os.path.isfile(folder):
                    raise EssentiaError('Cannot create directory "{}". '
                                        'There is a file with the same name. '
                                        'Aborting analysis.'.format(folder))
                else:
                    os.makedirs(folder, exist_ok=True)

                cmd_lines.append(extractor_cmd + [audio_file, out_file])
                ids_to_compute.append(filename)
            pbar.update()
    pbar.close()

    print(f'preanalysis done. There are ({len(cmd_lines)}) spectrograms')
    target = min(target, len(cmd_lines))

    print(f'targeting ({target}) spectrograms')
    cmd_lines = cmd_lines[:target]
    ids_to_compute = set(ids_to_compute[:target])

    ids = list(already_computed.union(ids_to_compute))
    if not whitelist:
        with open('ids', 'w') as f:
            f.write('\n'.join(ids))

    # analyze
    log_lines = []
    total, errors, oks = 0, 0, 0
    if cmd_lines:
        p = Pool(jobs)
        outs = p.map(partial(_subprocess, verbose=verbose), cmd_lines)

        total = len(outs)
        status, cmd, stderr = zip(*outs)

        oks, errors = 0, 0
        for i, cmd_idx, err in zip(status, cmd, stderr):
            if i == 0:
                oks += 1
                log_lines.append('"{}" ok!'.format(cmd_idx))
            else:
                errors += 1
                log_lines.append('"{}" failed'.format(cmd_idx))
                log_lines.append('  "{}"'.format(err))

    summary = ("Analysis done for {} files. {} files have been skipped due to errors, "
               "{} were successfully processed and {} already existed.\n").format(total, errors, oks, skipped_count)
    print(summary)

    # generate log
    if generate_log:
        log = [summary] + log_lines

        with open(os.path.join(output_dir, 'log'), 'w') as f:
            f.write('\n'.join(log))

def main(audio_dir, output_dir, generate_log=True, verbose=True,
         audio_types='mp4', skip_analyzed=True, jobs=0, blacklist_file=None,
         whitelist_file=None, target=1000000):
    """Generates mel bands for every audio file matching `audio_types` in `audio_dir`.
    The generated .npy files are stored in `output_dir` matching the folder
    structure found in `audio_dir`.
    """

    extractor_cmd = [sys.executable, os.path.join(os.path.dirname(__file__),
                                                  'melspectrogram.py')]

    # Set --force as a hardcoded flat.
    # Use skip_analyzed to control this behavior.
    extractor_cmd += ['--force']

    _batch_extractor(audio_dir, output_dir, extractor_cmd, 'mmap',
                     generate_log=generate_log, audio_types=audio_types,
                     skip_analyzed=skip_analyzed, jobs=jobs, verbose=verbose,
                     blacklist_file=blacklist_file, whitelist_file=whitelist_file,
                     target=target)



if __name__ == '__main__':
    parser = ArgumentParser(
        description='Computes the mel spectrogram of a given audio file.')
    parser.add_argument('audio_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--audio_types', default='mp4')
    parser.add_argument('--blacklist-file', default=None, 'files to skip')
    parser.add_argument('--whitelist-file', default=None, 'files to process exclusively')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--target', type=int, default=1000000, 'maximum number of spectrograms to compute')

    main(**vars(parser.parse_args()))
