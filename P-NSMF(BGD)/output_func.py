import csv


def create_result_file(result_file, args):
    with open(result_file, "a", newline="") as f:
        f.write(args + "\n\n")
        f.write("iter,pre,rec,F1,NDCG,oneCall \n")


def write_result(result_file, res_row):
    with open(result_file, "a", newline="") as f:
        csv_writer = csv.writer(f, delimiter="\t")
        csv_writer.writerow(res_row)
