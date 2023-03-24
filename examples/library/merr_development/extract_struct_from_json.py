import json
from pymatgen.core.structure import Structure

if __name__ == "__main__":
            filepath = "JSON/Volume_BCC/BCC_2.9.json"
            with open(filepath) as file:
                file.readline()
                try:
                    data = json.loads(file.read(), parse_constant=True)
                except Exception as e:
                    print("Trouble Parsing Json Data: ", filepath)
                assert len(data) == 1, "More than one object (dataset) is in this file"
                data = data['Dataset']
                assert len(data['Data']) == 1, "More than one configuration in this dataset"

                data['Group'] = filepath.split("/")[-2]
                data['File'] = filepath.split("/")[-1]
                assert all(k not in data for k in data["Data"][0].keys()), "Duplicate keys in dataset and data"
                struct_data = data['Data'][0]
                data.update(data.pop('Data')[0])  # Move data up one level
            struct = Structure(lattice=struct_data['Lattice'],coords=struct_data['Positions'],species=struct_data['AtomTypes'],coords_are_cartesian=True)
            assert struct.is_valid()
            print (struct)
