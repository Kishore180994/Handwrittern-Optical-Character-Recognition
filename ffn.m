function [out_a1] = ffn(weights1,weights2,b1,b2,input,no_of_input_nodes,no_of_samples,no_of_hidden_nodes,no_of_output_nodes)
a(no_of_input_nodes)=0;
a1(no_of_input_nodes)=0;
error(1:no_of_output_nodes)=1;
weights = 0;
out_a(no_of_input_nodes)=0;
lazyness=0;
inputs=1;
p=0.5;
for m=1:no_of_samples
    for k=1:no_of_input_nodes
        x(k)=input(k,m);
    end
    %Feed-Forward Network
    for hidden_nodes = 1: no_of_hidden_nodes
        a(hidden_nodes)=0;
        for input_nodes=1:no_of_input_nodes
            a(hidden_nodes)=a(hidden_nodes)+ x(input_nodes)*weights1(input_nodes,hidden_nodes);
        end
        a(hidden_nodes)=a(hidden_nodes)+b1(hidden_nodes);
        out_a(hidden_nodes) = sigmoid(a(hidden_nodes)*0.5);
    end
    for output_nodes = 1: no_of_output_nodes
        a1(output_nodes)=0;
        for hidden_nodes = 1: no_of_hidden_nodes
            a1(output_nodes) = a1(output_nodes) + out_a(hidden_nodes)* weights2(hidden_nodes,output_nodes);
        end
       a1(output_nodes) = a1(output_nodes) + b2(output_nodes);
        out_a1(output_nodes) = sigmoid(a1(output_nodes)*0.5);
         out_a1(output_nodes) = round(out_a1(output_nodes));
    end
    end
end