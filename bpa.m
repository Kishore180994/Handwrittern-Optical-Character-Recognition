function [w1,w2,b1,b2] = bpa(input,no_of_input_nodes,no_of_samples,no_of_hidden_nodes,no_of_output_nodes,target)
% w1 = -1 + (1-(-1)*rand(no_of_input_nodes,no_of_hidden_nodes));
INIT_EPSILON = 0.5;
w1=rand(no_of_input_nodes,no_of_hidden_nodes)*(2 * INIT_EPSILON) - INIT_EPSILON;
% w2 = -1 + (1-(-1)*rand(no_of_hidden_nodes,no_of_output_nodes));
w2=rand(no_of_hidden_nodes,no_of_output_nodes)*(2 * INIT_EPSILON) - INIT_EPSILON;
b1 =  -1 + (1-(-1)*rand(1,no_of_hidden_nodes));
b2 =  -1 + (1-(-1)*rand(1,no_of_output_nodes));
a(no_of_input_nodes)=0;
a1(no_of_input_nodes)=0;
error(1:no_of_output_nodes)=1;
err=1;
weights = 0;
out_a(no_of_input_nodes)=0;
lazyness=0;
inputs=1;
p=0.5;
input(:,1);
while any (err>0.5)
   for abc=1:9999
   lazyness=lazyness+1;
    for m=1:no_of_samples
    for k=1:no_of_input_nodes
        x(k)=input(k,m);
    end
    for k=1:no_of_output_nodes
        d(k)=target(k,m);
    end
    %Feed-Forward Network
    for hidden_nodes = 1: no_of_hidden_nodes
        a(hidden_nodes)=0;
        for input_nodes=1:no_of_input_nodes
            a(hidden_nodes)=a(hidden_nodes)+ x(input_nodes)*w1(input_nodes,hidden_nodes);
        end
        a(hidden_nodes)=a(hidden_nodes)+b1(hidden_nodes);
        out_a(hidden_nodes) = sigmoid(a(hidden_nodes)*0.5);
    end

    for output_nodes = 1: no_of_output_nodes
        a1(output_nodes)=0;
        for hidden_nodes = 1: no_of_hidden_nodes
            a1(output_nodes) = a1(output_nodes) + out_a(hidden_nodes)* w2(hidden_nodes,output_nodes);
        end
       a1(output_nodes) = a1(output_nodes) + b2(output_nodes);
        out_a1(output_nodes) = sigmoid(a1(output_nodes)*0.5);
        %Error calculations
        error(output_nodes) = d(output_nodes)-out_a1(output_nodes);%%%%%%%%%
        err(output_nodes) = (error(output_nodes).*error(output_nodes))/2.0;
        err = sum(err)/output_nodes;
        error1(output_nodes) = error(output_nodes)* out_a1(output_nodes)*(1-out_a1(output_nodes));
    end
    %%FeedForward network complete
    
    %%BackPropagation Starts
    for output_nodes = 1: no_of_output_nodes
        for hidden_nodes = 1: no_of_hidden_nodes
            delta_w2(hidden_nodes,output_nodes) = p*error1(output_nodes)*out_a(hidden_nodes);
            w2(hidden_nodes,output_nodes) = w2(hidden_nodes,output_nodes)+ delta_w2(hidden_nodes,output_nodes);
        end
        delta_b(output_nodes) = p*error1(output_nodes)*b2(output_nodes);
        b2(output_nodes) = b2(output_nodes)+delta_b(output_nodes);
    end
    
    for hidden_nodes = 1: no_of_hidden_nodes
        weights=0;
        for output_nodes = 1:no_of_output_nodes
            weights = weights + w2(hidden_nodes,output_nodes)*error1(output_nodes);
        end
        error_a1(hidden_nodes) = (out_a(hidden_nodes) * (1 - out_a(hidden_nodes))) * weights;
    end
    
    
    for input_nodes = 1: no_of_input_nodes
        for hidden_nodes = 1: no_of_hidden_nodes
           delta_w1(input_nodes,hidden_nodes) = p*error_a1(hidden_nodes)*x(input_nodes);
           w1(input_nodes,hidden_nodes) = w1(input_nodes,hidden_nodes)+delta_w1(input_nodes,hidden_nodes);
        end
    end
    
    for hidden_nodes = 1:no_of_hidden_nodes
           delta_b1(hidden_nodes) = p*error_a1(hidden_nodes)*b1(hidden_nodes);
           b1(hidden_nodes) = b1(hidden_nodes)+delta_b1(hidden_nodes);
    end
    end
end
end
end