import struct
import re
import pandas

if __name__ == "__main__":
    INPUT_FILE = r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_22\Whole Cell +1V7200s 1sdischarge 0V2hrshort 8.21 overnight.nox"
    OUTPUT_FILE = r'C:\Users\user\Downloads\8.21overnight.xlsx'
    NUMBER_OF_DATA_POINTS = 0

    with open(INPUT_FILE, 'rb') as f:
        contents = f.read()

        data = []
        matches = list(re.finditer(b"[^\x00][\x00\x01]\x00[\x00\x3A][^\x00]\x00\x00\x06", contents))
        for match in matches:
            start = match.start()

            end = 0
            if not NUMBER_OF_DATA_POINTS:
                next_match = re.match(b'[^\x00]\x00\x00\x00[^\x00]\x00\x00\x06', contents[start + 8:])
                print(next_match.start() if next_match else -1, contents.find(b'\x00' * 200, start + 8))

                zero_count = 1
                begin = contents.find(b'\x00' * 800, start + 8)
                while contents[begin + zero_count * 8:begin + zero_count*8 + 8] == b'\x00'*8:
                    zero_count += 1
                print(zero_count)

                end = min(start + 8 + (next_match.start() if next_match else float('inf')), contents.find(b'\x00' * 800, start + 8))
            else:
                end = start + NUMBER_OF_DATA_POINTS * 8 + 8
            data.append([struct.unpack('d', doublebytes)[0] for doublebytes in [contents[j:j + 8] for j in range(start + 8, end, 8)]])

        print([len(d) for d in data])

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

        # [Record Signals(time, current, pxv, wev), OCP10Hz(time, wev, pxv, current), {OCP1Hz}] * 3, [LSV(Voltage applied)] * 3, [LSV(time, current, potential, voltage)]
        # Group into trials and sort
        data = data[:36] + data[39:]
        data = [data[i:i + 4] for i in range(0, len(data), 4)]
        data.sort(key=lambda x: x[0][0])
        data = [sublist for matrix in data for sublist in matrix]

        from openpyxl import Workbook

        # Create a new Excel workbook and select the active sheet
        wb = Workbook()
        ws = wb.active

        # Define your column headers
        headers = (
                          ['Time (s)', 'WE(1).Current (A)', 'pX(1).Voltage (V)', 'WE(1).Potential (V)'] +
                          ['Time (s)', 'WE(1).Current (A)', 'WE(1).Potential (V)', 'pX(1).Voltage (V)'] +
                          ['Time (s)', 'WE(1).Potential (V)', 'pX(1).Voltage (V)', 'WE(1).Current (A)'] * 2
                  ) * (len(data) // 16)
        ws.append(headers)

        print(len(data))
        print([len(d) for d in data])

        # Write data rows
        for i in range(max(len(d) for d in data)):
            row = [str(d[i]) if i < len(d) else '' for d in data]
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