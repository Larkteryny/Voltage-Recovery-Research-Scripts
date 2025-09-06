import struct
import re
from collections import Counter

import pandas
import numpy as np

if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_04\Alternating Automated Voltage Recovery Whole Cell 0.5V with 50s 0V2hrshort 8.03 overnight.nox"#r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_05\Alternating Automated Voltage Recovery Whole Cell 1.5V with 50s 0V2hrshort 8.04 overnight.nox"#"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_22\Whole Cell +1V7200s 1sdischarge 0V2hrshort 8.21 overnight.nox"
    OUTPUT_FILE = r'C:\Users\user\Downloads\8.03overnight_unsorted.xlsx'#'C:\Users\user\Downloads\8.21overnight.xlsx'
    NUMBER_OF_DATA_POINTS = 0

    with open(INPUT_FILE, 'rb') as f:
        contents = f.read()

        data = []
        #AD0000E000000006  0.5V 410
        #3C01006003000006  1.5V 615
        #D601003A02000006
        #1F00000002000006 1.5V
        #1F0000DC05000006 1.5V
        #3F01006003000006 1.5V 615
        #1D00000008000006 1.5V adc164
        #1E0000DC05000006 1.5V time since start for 10Hz OCP? followed by probably offset data for 743 rows
        #3C0100B003000006 1.5V positive LSV
        print(list(re.finditer(b"[^\x00][\x00\x01]\x00[\x00\x3A\x60\xB0][^\x00]\x00\x00\x06", b'\x3C\x01\x00\xB0\x03\x00\x00\x06')))
        matches = list(re.finditer(b"[^\x00][\x00\x01]\x00[\x00\x3A\x60\xB0][^\x00]\x00\x00\x06", contents)) + list(re.finditer(b"[^\x00]\x00\x00[\xE0-\xFF]\x00\x00\x00\x06", contents))
        for match in matches:
            start = match.start()

            end = 0
            if not NUMBER_OF_DATA_POINTS:
                next_match = re.match(b'[^\x00]\x00\x00\x00[^\x00]\x00\x00\x06', contents[start + 8:])
                print(next_match.start() if next_match else -1, contents.find(b'\x00' * 200, start + 8))

                zero_count = 1
                begin = contents.find(b'\x00' * 150, start + 8)
                while contents[begin + zero_count * 8:begin + zero_count*8 + 8] == b'\x00'*8:
                    zero_count += 1
                print(zero_count)

                end = min(start + 8 + (next_match.start() if next_match else float('inf')), contents.find(b'\x00' * 150, start + 8))
            else:
                end = start + NUMBER_OF_DATA_POINTS * 8 + 8
            data.append([struct.unpack('d', doublebytes)[0] for doublebytes in [contents[j:j + 8] for j in range(start + 8, end, 8)]])

        print([len(d) for d in data])
        print(len(data))
        from matplotlib import pyplot as plt
        #for x in data:
        #    plt.plot(x)
        #    plt.show()

        """peaks = [[], [], [], []]
        for group in range(0, len(data), 5):
            if group % 2:
                peaks[2].append(max(data[group + 1]))
                peaks[3].append(max(data[group + 3]))
            else:
                peaks[0].append(min(data[group + 1]))
                peaks[1].append(min(data[group + 3]))
        print(*peaks,sep='\n')

        import sys
        sys.exit()"""
        # Find mode array which is Potential Applied (V) data
        '''data_discretized = [tuple([abs(y) for y in x[:20]]) for x in data]
        from collections import Counter
        mode = Counter(data_discretized).most_common(1)[0]
        print(mode)
        data = [x for x in data if tuple([abs(y) for y in x[:20]]) != mode[0]]'''
        # Delete columns with near constant 0.00244 step
        def prox_count(a, tgt1, tgt2, tol):
            return sum(abs(abs(x) - tgt1) < tol or abs(abs(x) - tgt2) < tol for x in a)
        data = [x for x in data if not(prox_count(np.diff(x), 0.00244, 0.00245, 0.000002)/len(x) > 0.99 and prox_count([round(x[0], 1) - x[0]], 0.00250, 0.00250, 0.00001))]
        # Delete mistaken columns with unique length
        len_counts = Counter([len(x) for x in data])
        data = [x for x in data if len_counts[len(x)] > 1]
        print([len(d) for d in data])
        print(len(data))
        print(data[0])
        # [Record Signals(time, current, pxv, wev), OCP10Hz(time, wev, pxv, current), {OCP1Hz}] * 3, [LSV(Voltage applied)] * 3, [LSV(time, current, potential, voltage)]
        # Group into trials and sort
        data = [data[i:i + 4] for i in range(0, len(data), 4)]
        data.sort(key=lambda x: x[0][0])
        for i in range(len(data) - 1, 0, -1):
            if data[i][0][0] == data[i - 1][0][0]: data.pop(i)  # remove duplicates
            elif len(data[i-1][0]) == 205: print(data[i-1][0][0])
        time, pot = [t - 0*7000*(i//3) for i, x in enumerate(data) for t in x[0]], [x[1] if len(x[0]) in [1500, 300] else x[2] for x in data]
        pot = [p for x in pot for p in x]
        plt.scatter(time, pot, s=2)
        plt.show()
        data = [sublist for matrix in data for sublist in matrix]

        print(list([prox_count(np.diff(x), 0.00244, 0.00245, 0.000002)/len(x), prox_count([round(x[0], 1) - x[0]], 0.00250, 0.00250, 0.00001)] for x in data))
        print(list(round(x[0], 1) - x[0] for x in data))
        #print(np.diff(data[0]))

        from openpyxl import Workbook

        # Create a new Excel workbook and select the active sheet
        wb = Workbook()
        ws = wb.active

        # Define your column headers
        headers = (
                          ['Time (s)', 'WE(1).Current (A)', 'WE(1).Potential (V)', 'pX(1).Voltage (V)'] +
                          ['Time (s)', 'WE(1).Potential (V)', 'pX(1).Voltage (V)', 'WE(1).Current (A)'] * 2
                  ) * (len(data) // 12)
        ws.append(headers)

        print(len(data))
        print([len(d) for d in data])

        # Write data rows
        for i in range(max(len(d) for d in data)):
            row = [d[i] if i < len(d) else '' for d in data]
            ws.append(row)

        # Save the workbook
        wb.save(OUTPUT_FILE)
        import sys
        sys.exit()

        # Write extracted data
        with open(OUTPUT_FILE, 'w') as o:
            import sys
            sys.exit()
            o.write(
                ('Time (s),WE(1).Current (A),pX(1).Voltage (V),WE(1).Potential (V),'
                 + 'Time (s),WE(1).Current (A),WE(1).Potential (V),pX(1).Voltage (V),'
                 + 'Time (s),WE(1).Potential (V),pX(1).Voltage (V),WE(1).Current (A),' * 2)
                * (len(data) // 4) + '\n')

            print(len(data))
            print([len(d) for d in data])

            for i in range(max(len(d) for d in data)):
                line = ','.join(str(d[i]) if i < len(d) else '' for d in data)
                o.write(line + '\n')