import json

def dumpl(list:list):
    return json.dumps(list)+"\n"

def read(f):
    return [json.loads(line) for line in f]

if __name__ == "__main__":
    
    data = [[1,2,3], [4,5,6], [7, 6, 8]]
    with open('data.listl', 'a') as f:
        for d in data:
            f.write(dumpl(d))
            
    with open('data.listl', 'r') as f:
        data_loaded = read(f)
    
    print(data_loaded, type(data_loaded))