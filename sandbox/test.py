def main():
    string = ""
    for i in range(9, 37):
        string += fr"D:\Raymetrics_Tests\BOMA2025\20250923\output{i}.txt,"
    return string

if __name__ == "__main__":
    print(main())