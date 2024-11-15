import tarfile
import io
import csv

import pandas as pd

from paths import ACM_DATASET_FILE, DBLP_DATASET_FILE


def read_first_n_lines(tgz_filename, n=10):
    with tarfile.open(tgz_filename, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".txt"):
                with tar.extractfile(member) as file:
                    # Process the file content line by line
                    line_count = 0
                    for line in io.TextIOWrapper(file, encoding='utf-8'):
                        print(line.strip())
                        line_count += 1
                        if line_count == n:
                            break


def extract_publications(tgz_filename, output_csv_filename):
    with tarfile.open(tgz_filename, "r:gz") as tar, open(output_csv_filename, mode='w', newline='',
                                                         encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|')
        csv_writer.writerow(['PaperID', 'Title', 'Authors', 'Venue', 'Year'])

        current_entry = {}
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".txt"):
                with tar.extractfile(member) as file:
                    for line in io.TextIOWrapper(file, encoding='utf-8'):
                        line = line.strip()
                        if line.startswith("#*"):
                            current_entry['Title'] = line[2:]
                        elif line.startswith("#@"):
                            current_entry['Authors'] = line[2:]
                        elif line.startswith("#t"):
                            current_entry['Year'] = line[2:]
                        elif line.startswith("#c"):
                            current_entry['Venue'] = line[2:]
                        elif line.startswith("#index"):
                            current_entry['PaperID'] = line[6:]
                        elif line == "":
                            # Check if the entry meets the criteria before writing it to CSV
                            if 'Year' in current_entry and 1995 <= int(current_entry['Year']) <= 2004:
                                if 'Venue' in current_entry and any(
                                        keyword.lower() in current_entry['Venue'].lower() for keyword in
                                        ["sigmod", "vldb"]):
                                    csv_writer.writerow([current_entry.get(key, '') for key in
                                                         ['PaperID', 'Title', 'Authors', 'Venue', 'Year']])
                            # Reset current entry
                            current_entry = {}


def remove_duplicates(dataset_file):
    df = pd.read_csv(dataset_file, sep='|', skiprows=1, names=['paperId', 'title', 'authors', 'venue', 'year'])
    df.drop_duplicates(keep='first', ignore_index=True, inplace=True)
    df.to_csv(dataset_file, sep="|", index=False)

if __name__ == "__main__":
    extract_publications("citation-acm-v8.txt.tgz", ACM_DATASET_FILE)
    extract_publications("dblp.v8.tgz", DBLP_DATASET_FILE)
    
    remove_duplicates(ACM_DATASET_FILE)
    remove_duplicates(DBLP_DATASET_FILE)
