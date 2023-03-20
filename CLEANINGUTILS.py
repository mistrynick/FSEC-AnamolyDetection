import csv

# import datetime
# from matplotlib import pyplot as plt
import pandas as pd



def convertData(file_name):
    with open(file_name) as file:
        c = csv.reader(file)
        header = next(c)
        print(header)
        f = open("secondPart_11.csv", 'a', newline='')
        writer = csv.writer(f)
        dates = []

        for i, line in enumerate(file):

            row = line.split(',')
            # print(row)
            x = row[-1].replace('\n', '')
            del row[-1]
            row.append(x)

            # print("now")
            # print(row)

            if row[0] != "#NAME?":
                dates.append(row)

        for date in dates:
            # del date[-1]
            date.append("1")

            writer.writerow(date)


def getSomeDat(file_name):
    with open(file_name) as file:
        c = csv.reader(file)
        header = next(c)
        print(header)
        count = 0
        to_keep = []
        to_drop = []
        for item in header:
            if 31 < count < 42:
                to_keep.append(item)
            else:
                to_drop.append(item)
            count = count + 1

        print("to keep:")
        print(to_keep)
        print(len(to_keep))
        print("to drop")
        print(to_drop)
        return to_keep, to_drop


def foo():
    to_keep, to_drop = getSomeDat('FSEC_SLTE_SWTDI_data_2019_06_20_To_2022_06_28.csv')
    to_drop.append('Sys2_PF_Avg')
    to_drop.append('Sys2_VAR_Avg')
    to_keep.remove('Sys2_PF_Avg')
    to_keep.remove('Sys2_VAR_Avg')
    print("NOW\n\n\n")
    print(to_keep)
    print(len(to_keep))
    print(to_drop)

    df = pd.read_csv('FSEC_SLTE_SWTDI_data_2019_06_20_To_2022_06_28.csv', low_memory=False)
    df.drop(to_drop, inplace=True, axis=1)
    df.to_csv('sndPart1.csv', index=False)


def concat(file_name1, file_name2):
    with open(file_name1) as f1:
        with open(file_name2) as f2:
            c1 = csv.reader(f1)
            c2 = csv.reader(f2)
            newFile = open("final_data.csv", 'a', newline='')
            f_writer = csv.writer(newFile)
            for i, line in enumerate(f1):
                row = line.split(',')
                # print(row)
                x = row[-1].replace('\n', '')
                del row[-1]
                row.append(x)
                f_writer.writerow(row)
            for i, line in enumerate(f2):
                row = line.split(',')
                # print(row)
                x = row[-1].replace('\n', '')
                del row[-1]
                row.append(x)
                f_writer.writerow(row)
            newFile.close()


def writeAppend(file_name):
    with open("final_data.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        with open(file_name) as file1:
            for i, line in enumerate(file1):
                row = line.split(',')
                #print(row)
                #print(len(row))
                x = row[-1].replace('\n', '')
                del row[-1]
                row.append(x)
                writer.writerow(row)


def foo2():
    to_keep, to_drop = getSomeDat('FSEC_SLTE_SWTDI_data_2019_06_20_To_2022_06_28.csv')
    to_drop.append('Sys4_VAR_Avg')
    to_drop.append('Sys4_PF_Avg')
    df = pd.read_csv('FSEC_SLTE_SWTDI_data_2019_06_20_To_2022_06_28.csv', low_memory=False)
    df.drop(to_drop, inplace=True, axis=1)
    df.to_csv('dataToConcat_1.csv', index=False)
    convertData('dataToConcat_1.csv')


def main():
    print("asdf")
    to_keep, to_drop = getSomeDat('FSEC_SLTE_SWTDI_data_2019_06_20_To_2022_06_28.csv')



main()
