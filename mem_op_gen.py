
import os

def struct_gen(format_str, file):

    format = format_str.split(":")

    # Defining the required structs
    encoding = "".join(format)
    file.write(f"struct tile_{encoding}")
    file.write("{\n")
    for i in range(len(format)):
        if(format[i] == "s"):
            file.write(f"std::vector<int> pos{i + 1};\n")
            file.write(f"std::vector<int> crd{i + 1};\n")
    file.write("std::vector<float> vals;\n")
    file.write("};\n\n")

def mem_op_gen(format_str, file) :

    in_format_str, out_format_str = format_str.split("->")
    in_format = in_format_str.split(":")
    out_format = out_format_str.split(":")
    
    # Defining the required encodings
    in_encoding = "".join(in_format)
    out_encoding = "".join(out_format)

    # Defining the function to generate the memory operation
    file.write(f"tile_{out_encoding} mem_op_{in_encoding}_{out_encoding}(tile_{in_encoding} tensor_op, int index)")
    file.write("{\n")

    len_in = len(in_format)
    len_out = len(out_format)

    if out_format != in_format[len_in - len_out:]:
        print("Error: Output format should be a subset of input format")
        return

    for i in range(len_out):
        if(out_format[i] == "s"):
            file.write(f"int *pos{i + 1} = tensor_op.pos{i + 1 + len_in - len_out}.data();\n")
            file.write(f"int *crd{i + 1} = tensor_op.crd{i + 1 + len_in - len_out}.data();\n")

    file.write("float *vals = tensor_op.vals.data();\n\n")

    file.write(f"tile_{out_encoding} tile_op;\n\n")

    for i in range(len_out):
        file.write(f"int i{i}_end = 0;\n")


    for i in range(len_out):
        if i == 0:
            prev_i = "ndex"
        else:
            prev_i = i - 1

        if(out_format[i] == "d"):
            file.write(f"for(int i{i} = i{prev_i} * i{i}_dim; i{i} < (i{prev_i} + 1) * i{i}_dim; i{i}++)")
            file.write("{\n")
            file.write(f"i{i}_end = 0;\n")
            file.write(f"if(i{i} == ((i{prev_i} + 1) * i{i}_dim - 1)) i{i}_end = 1;\n")
        elif(out_format[i] == "s"):
            if i == 0: 
                file.write(f"tile_op.pos{i+1}.push_back(pos{i+1}[index]);\n")
                file.write(f"tile_op.pos{i+1}.push_back(pos{i+1}[index + 1]);\n")
            else: 
                file.write(f"tile_op.pos{i+1}.push_back(pos{i+1}[i{prev_i}]);\n")
                file.write("if(i0_end")
                for j in range(1, i):
                    file.write(f" && i{j}_end")
                file.write(") ")
                file.write(f"tile_op.pos{i+1}.push_back(pos{i+1}[i{i-1} + 1]);\n")
            file.write(f"for(int i{i} = pos{i+1}[i{prev_i}]; i{i} < pos{i+1}[i{prev_i} + 1]; i{i}++)")
            file.write("{\n")
            file.write(f"i{i}_end = 0;\n")
            file.write(f"if(i{i} == (pos{i+1}[i{prev_i} + 1] - 1)) i{i}_end = 1;")
            file.write(f"tile_op.crd{i+1}.push_back(crd{i+1}[i{i}]);\n")

        if(i == len_out - 1):
            file.write(f"tile_op.vals.push_back(vals[i{i}]);\n")
            file.write("}" * len_out)
            file.write("\n\n")

    for i in range(len_out):
        if(out_format[i] == "s"):
            file.write(f"int pos{i+1}_start = tile_op.pos{i+1}[0];\n")
            file.write(f"std::transform(tile_op.pos{i+1}.begin(), tile_op.pos{i+1}.end(), tile_op.pos{i+1}.begin(), [pos{i+1}_start](int elem)" + "{return elem - pos" +  str(i+1) + "_start; });\n")
            

    file.write("return tile_op;\n")
    file.write("}\n\n")

if __name__ == "__main__":

    in_format = ["s", "s", "s", "s", "d", "s"]
    out_format = ["s", "s", "d", "s"]
    file_name = "mem_op.cpp"
    mem_op_gen(in_format, out_format, file_name)
    os.system(f"clang-format -i {file_name}")



