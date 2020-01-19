function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  return y * (1 - y);
}

class NeuralNetworks {

  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;
    
    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);      //weights between input and hidden
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);      //weights between hidden and output
    this.weights_ih.randomize();
    this.weights_ho.randomize();
    
    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();
    this.learning_rate = 0.1;
  }
  
  feedforward(input_array) {
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    
    hidden.map(sigmoid);    //Activation Function(Sigmoid used right now)
    
    
    // Generating the Output's Outputs
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(sigmoid);
    
    // Sending back to the caller!
    return output.toArray();
  }
  
  train(input_array, target_array) {
    
    // Generating the Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    
    hidden.map(sigmoid);    //Activation Function(Sigmoid used right now)
    
    
    // Generating the Output's Outputs
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(sigmoid);
    
    //Convert array to matrix object
    //outputs = Matrix.fromArray(outputs);
    let targets = Matrix.fromArray(target_array);
    
    //calculate the errors
    // ERROR = TARGETS - OUTPUTS
    let output_errors = Matrix.subtract(targets, outputs);
    
    //let gradient = outputs * (1 - outputs);
    // calculate gradient
    let gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);
    
    //calculate deltas
    let hidden_T = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);
    
    // weight_ho_deltas = learning_rate * output_errors * outputs * (1 - outputs) * hidden_T
    
    //adjust the weights by its deltas
    this.weights_ho.add(weight_ho_deltas);
    //adjust the bias by its deltas(which is gradients)
    this.bias_o.add(gradients);
    
    //calculate the hidden layer errors
    let weights_ho_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(weights_ho_t, output_errors);
    
    // calculate hidden gradient
    let hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);
    
    // calculate input->hidden deltas
    let input_T = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradient, input_T);
    
    // weight_hi_deltas = learning_rate * hidden_errors * hidden * (1 - hidden) * input_T
    
    this.weights_ih.add(weight_ih_deltas);
    //adjust the bias by its deltas(which is gradients)
    this.bias_h.add(hidden_gradient);

    
    //outputs.print();
    //targets.print();
    //output_errors.print();
    
  }
  
}
