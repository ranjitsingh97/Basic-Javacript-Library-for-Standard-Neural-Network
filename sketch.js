let tarining_data = [
  {
     inputs : [0, 1],
     targets : [1]
  },
  {
     inputs : [1, 0],
     targets : [1]
  },
  {
     inputs : [0, 0],
     targets : [0]
  },
  {
     inputs : [1, 1],
     targets : [0]
  }
];

function setup() {
  
  let nn = new NeuralNetworks(2, 2, 1);
  
  for(let i=0; i<50000; i++) {
    let data = random(tarining_data);
    nn.train(data.inputs, data.targets);
  }
  
  console.log(nn.feedforward([1,0]));
  console.log(nn.feedforward([0,1]));
  console.log(nn.feedforward([0,0]));
  console.log(nn.feedforward([1,1]));
  
  //let inputs = [1, 0];
  //let targets = [1, 0];
  
  ////let output = nn.feedforward(inputs);
  
  //nn.train(inputs, targets);
  ////console.log(output);
  
}
