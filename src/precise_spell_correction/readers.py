import csv
import json
from pathlib import Path

from loguru import logger


class CsvIO:

    def __init__(self, text_file, header=None, encoding='utf8'):
        self.current_line = None
        self.fh = open(text_file, newline='', encoding=encoding)
        self.reader = csv.DictReader(self.fh)
        if header and header in self.reader.fieldnames:
            pass
        else:
            for name in ['text', 'note_text']:
                if name in self.reader.fieldnames:
                    header = name
                    logger.info(f'Using text column: {header}')
                    break
            if header is None:
                raise ValueError(f'Cannot determine text column in CSV file: {text_file}')
        self.header = header
        self.reader_iter = iter(self.reader)
        self.outfh = open(f'{text_file.stem}.sc.csv', 'w', newline='', encoding=encoding)
        self.writer = csv.DictWriter(self.outfh, fieldnames=list(self.reader.fieldnames) + ['new_text'])
        self.writer.writeheader()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def __iter__(self):
        return self

    def __next__(self):
        self.current_line = next(self.reader_iter)
        return self.current_line[self.header]

    def write(self, new_text):
        self.current_line['new_text'] = new_text
        self.writer.writerow(self.current_line)

    def close(self):
        self.fh.close()
        self.outfh.close()


class JsonlIO:

    def __init__(self, text_file, header=None, encoding='utf8'):
        self.fh = open(text_file, encoding=encoding)
        self.current_line = json.loads(next(self.fh).strip())
        self.is_first_line = True
        if header and header in self.current_line.keys():
            pass
        else:
            for name in ['text', 'note_text']:
                if name in self.current_line.keys():
                    header = name
                    logger.info(f'Using text column: {header}')
                    break
            if header is None:
                raise ValueError(f'Cannot determine text column in JSONL file: {text_file}')
        self.header = header
        self.outfh = open(f'{text_file.stem}.sc.jsonl', 'w', newline='')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_first_line:
            self.is_first_line = False
            return self.current_line[self.header]
        self.current_line = json.loads(next(self.fh).strip())
        return self.current_line[self.header]

    def write(self, new_text):
        self.current_line['new_text'] = new_text
        self.outfh.write(json.dumps(self.current_line) + '\n')

    def close(self):
        self.fh.close()
        self.outfh.close()


class TextIO:

    def __init__(self, text_file, header=None, encoding='utf8'):
        with open(text_file, encoding=encoding) as fh:
            self.text = fh.read()
        self.outfh = open(f'{text_file.stem}.sc.txt', 'w')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.text is None:
            raise StopIteration
        text = self.text
        self.text = None
        return text

    def write(self, new_text):
        self.outfh.write(new_text)

    def close(self):
        self.outfh.close()


class DirIO:

    def __init__(self, text_file: Path, header=None, encoding='utf8'):
        self.outdir = text_file.parent / f'{text_file.stem}.sc'
        self.outdir.mkdir()
        self.current_filename = None
        self.encoding = encoding
        if header:
            self.iter = self.text_file.glob(header)
        else:
            self.iter = self.text_file.iterdir()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def __iter__(self):
        return self

    def __next__(self):
        path = next(self.iter)
        self.current_filename = path.stem
        with open(path, encoding=self.encoding) as fh:
            text = fh.read()
        return text

    def write(self, new_text):
        with open(self.outdir / self.current_filename, 'w', encoding=self.encoding) as out:
            out.write(new_text)

    def close(self):
        pass
