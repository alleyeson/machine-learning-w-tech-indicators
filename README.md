# machine-learning-w-tech-indicators
This project was a school project with combines machine learning in the form SVM and neural networks with technical indicators

#code in c++
##source Dave Miller code Neural Networks: http://www.millermattson.com/dave/?p=201 

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>


using namespace std;

struct Connection
{
    double weight; //weight between layers
    double deltaWeight; // to store prior change so as to add to new change to it
};


class Neuron;
class Bias_Neuron; // not yet in use

typedef vector<Neuron> Layer; // Layers a vector Network of Neurons

// ****************** class Neuron ******************
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex, vector<double> prior_weight); //Neuron decleration
    void setOutputVal(double val) { m_outputVal = val; } //set the output val, forms trickle effect later on
    double getOutputVal(void)   { return m_outputVal; } //function to get each Neurons output
    void feedForward(  Layer &prevLayer); //to push in the output from previous layer into current layer
    void calcOutputGradients(double targetVal); //gradient calculation to learn from output layer
    void calcHiddenGradients(  Layer &nextLayer); // same as above but for hidden layer
    void updateInputWeights(Layer &prevLayer); // updates weights from previous layer
    vector<double> past_error_weights;
    
private:
    static double eta;   // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(  Layer &nextLayer)  ;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
    double m_hidden_gradient;
    double m_output_gradient;
    double delta_error;
    double weight_o;
};

double Neuron::eta = 0.5;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]


void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    //collect wrong weights in neuron rather than in net. Forgot the time for now! If neuron/net knows the weight it is wrong then append.
    
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];
        //eta = delta_error/((m_output_gradient*m_output_gradient) + (m_hidden_gradient*m_hidden_gradient));
        
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        
        // Individual input, magnified by the gradient and train rate:
        // Also add momentum = a fraction of the previous delta weight;
        double newDeltaWeight =
        eta
        * neuron.getOutputVal()
        * m_gradient
        + alpha
        * oldDeltaWeight;
        
        //simple if function to collect wrong direction
        if (delta_error > 0.005) { //PUT PSEUDO GRADIENT DECENT SIMULATION HERE when ready
            weight_o = neuron.m_outputWeights[m_myIndex].weight;
            //check.push_back(myWeight);
            past_error_weights.push_back(weight_o); //collect error weights
            //  *** FING WAY TO GET EACH ELEMENT IN PAST ERROR, ELEMENT BY ELEMENT
            //cont_error_w.push_back(weight_o);
            //cout <<weight_o << endl;
            //cout << "non vect: " << myWeight << endl;
            //cout << past_error_weights.at(n) << endl;
        }
        
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
        
        //write if statement that depends on m_myIndex. If equals to element in topology (then it is last neuron, update for next neuron should depend on past error deltaweights and weights themselves
    }
}

double Neuron::sumDOW(  Layer &nextLayer)
{
    double sum = 0.0;
    
    // Sum our contributions of the errors at the nodes we feed.
    
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    
    return sum;
}

void Neuron::calcHiddenGradients(  Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
    m_hidden_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
    
}

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    delta_error = delta*delta;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
    m_output_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}

double Neuron::activationFunction(double x)
{
    // tanh - output range [-1.0..1.0]
    
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

void Neuron::feedForward(  Layer &prevLayer)
{
    double sum = 0.0;
    
    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
        prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    
    m_outputVal = Neuron::activationFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex, vector<double> prior_weight)
{
    
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        double dum_weight = randomWeight();
        //m_outputWeights.back().weight = randomWeight();
        for(int i=0;i<prior_weight.size();i++){
            double curVal= prior_weight.at(i);
            double diff = curVal-dum_weight;
            if(abs(diff)<0.005) // Rerand until weights not within 0.05 of parents ERROR weights
            { dum_weight = randomWeight();}
            else{m_outputWeights.back().weight = dum_weight; }
        }
        //cout<< "Weight connected" << endl;
    }
    
    m_myIndex = myIndex; // to know what weights to change specifically.
}


// ****************** class Net ******************
class Net
{
public:
    Net(  vector<unsigned> &topology);
    void feedForward(  vector<double> &inputVals);
    void backProp(  vector<double> &targetVals);
    void getResults(vector<double> &resultVals)  ;
    double getRecentAverageError(void)   { return m_recentAverageError; }
    void read_past_weights(vector<double> past_weight);
    
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
    vector<double> read_weight;
};


double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::read_past_weights(vector<double> past_weight)
{
    for (int c = 0; c < past_weight.size(); c++) {
        read_weight.push_back(past_weight.at(c)); //push in past wrong weights to detemine direction no to go
        //cout << read_weight.at(c);
    }
}

void Net::getResults(vector<double> &resultVals)
{
    resultVals.clear();
    
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());  //to results from net for prediction and or check
    }
}

void Net::backProp(  vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)
    
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
        //cout << "Target Value: " << targetVals[n] << endl;
        //cout << "Current Guess: " << outputLayer[n].getOutputVal() << endl;
        //cout << targetVals[n]  << " " << outputLayer[n].getOutputVal() << endl;
    }
    //m_error /= outputLayer.size() - 1; // get average error squared
    // double avg_error = 0.0;
    
    //cout << "Percentage error: " <<  m_error *100 << "%" << endl;
    m_error = sqrt(m_error); // RMS
    
    //cout << "Square root Percentage error: " << m_error*100 <<"%" << endl;
    
    //cout << m_error << endl;
    //avg_error += m_error/
    // Implement a recent average measurement
    
    m_recentAverageError =
    (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
    / (m_recentAverageSmoothingFactor + 1.0);  // for cumulative average error
    
    // /**** cout << m_recentAverageError << endl; //MOST IMPORTANT CHECK RIGHT HERE
    // cout << "Recent Average Error: " << m_recentAverageError*100 << endl << endl;
    // Calculate output layer gradients
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]); // calculat gradient for current network output layer
    }
    
    // Calculate hidden layer gradients
    
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);  // input layer from next layer
        }
    }
    
    // For all layers from outputs to first hidden layer,
    // update connection weights
    
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(  vector<double> &inputVals)
{
    //cout << "Size of input layer: " << m_layers[0].size() << endl;
    assert(inputVals.size() == m_layers[0].size()-1);
    
    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}
// MAKE VECTOR OF TYPE NET
Net::Net(  vector<unsigned> &topology)
{
    unsigned numLayers = topology.size(); // Ex: 1-3-1 one input layer, 3 neurons in hidden layer and one in output layer.
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer()); // pushback number of layers wanted
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        
        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum, read_weight)); // pushback neurons in each layer
            
            
            //cout << "Frame number: " << topology[layerNum]<< endl;
            //cout << "Made a Neuron!" << endl;
        }
        
        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().setOutputVal(1.0); //this must be changed later
    }
}

double funct(double x, double y){
    double distance = y-x;
    double prod = distance * distance;
    return prod;
}

double moving_average (vector<double> &y, double n)
{
    
    double moving_avg  = 0;
    double t = y.size()-1;
    double sum = 0;
    //double n = n;
    if (n> t) {
        return 9;
    }
    for (double i = t; i > t - n; --i) {
        //cout << "i: " << i << endl;
        //cout << y.at(i) << endl;
        sum += y.at(i);
        //cout << "sum: " << sum << endl;
        
    }
    moving_avg = sum/n;
    
    return moving_avg;
}

double weight_moving_avg (vector<double> &z, double n)
{
    double weight_moving_avg  = 0;
    double t = z.size()-1;
    double sum = 0;
    double denom_sum = 0;
    //double n = 2;
    //double n = n;
    for (double i = 0; i < n; i++) {
        
        //cout << "i: " << i << endl;
        //cout << y.at(t-i) << endl;
        sum += z.at(t-i)*(n-i);
        denom_sum += (n-i);
        
        //i+=i;
        //cout << "sum: " << sum << endl;
        
    }
    if (denom_sum == 0 ) {
        denom_sum = 1;
    } else {denom_sum = denom_sum; }
    weight_moving_avg = sum/denom_sum;
    
    return weight_moving_avg;
}

double momentum (vector<double> &z, double n)
{
    double t = z.size() - 1;
    double diff = z.at(t) - z.at(t-n);
    
    return diff;
}

double ret_min(vector<double> &x) {
    
    double min = 0;
    min = x.at(0);
    for (int i = 0; i< x.size()-1; i++) {
        //cout << i;
        if (min <= x.at(i+1)) {
            min = min;
            //cout << min;
        } else {min = x.at(i+1); }
        
    }
    return min;
}

double ret_max(vector<double> &x) {
    
    double max = 0;
    max = x.at(0);
    for (int i = 0; i< x.size()-1; i++) {
        //cout << i;
        if (max >= x.at(i+1)) {
            max = max;
            //cout << min;
        } else {max = x.at(i+1); }
        
    }
    return max;
}

double stochastic_k (vector<double> &y, double n)
{
    double stoch = 0;
    
    double lowest_value = ret_min(y);
    double largest_value = ret_max(y);
    
    double t = y.size() - 1;
    
    double num = y.at(t) - lowest_value;
    double denum = largest_value - lowest_value;
    
   // cout <<"yt: " << y.at(t) << endl;
   // cout << "largest val: " << largest_value << endl;
    //cout << "smallest: " << lowest_value << endl;
   // cout << "denum: " << denum << endl;
  //  cout << "num; " << num<< endl;
    
    if (denum == 0) {
        denum = 1;
    } else {denum = denum; }
    
    stoch = (num/denum)*100;
    
   // cout << "stoch: "<<  stoch << endl << endl;
    return stoch;
}

double stochastic_d (vector<double> &y, double n)
{
    double stoch_d = 0;
    double sum = 0;
    //double t = y.size() - 1;
    for (int i = 0; i < n-1 ; i++) {
        
        sum += stochastic_k(y, i); //increasing lag
    }
    
    stoch_d = sum/n;
    return stoch_d;
}

double accum_dist (vector<double> &d, double n){
    
    double accum = 0;
    
    double h = ret_max(d);
    double l = ret_min(d);
    double t_o = d.size() - 1;
    double d_t = d.at(t_o-1);
    
    double num = h - d_t;
    double denum = h-l;
    
    if (denum ==0 ) {
        denum = 1;
    } else {denum = denum; }
    accum = (num/denum);
    
    return accum;
}

double larry_william (vector<double>&r , double n){
    
    double will = 0;
    vector<double> dum;
    
    for (double t = n; t>0; --t) {
        dum.push_back(r.at(t));
    }
    
    double h_n = ret_max(dum);
    double l_n = ret_min(dum);
    double r_t = r.at(r.size()-1);
    
    double num = h_n - r_t;
    double denum = h_n - l_n;
    
    if (denum == 0) {
        denum = 1;
    } else {denum = denum;}
    
    will = (num/denum)*100;
    
    return will;
}

double RSI (vector<double> &x, double n)
{
    double rsi = 0;
    
    double ups_n = 0;
    double downs_n = 0;
    double diff = 0;
    
    for (double i = 0 ; i< x.size()-1; i++) {
        diff = x.at(i+1)- x.at(i);
        if (diff > 0 ) {
            ups_n += diff;
        } else {downs_n += diff; }
    }
    ups_n = ups_n/n;
    downs_n = downs_n/n;
    
    double ratio = 1;
    if (downs_n == 0) {
        ratio = 0;
    }else {ratio = ups_n/downs_n;}
    //double ratio = ups_n/downs_n;
    double denum = (1+ratio);
    
    rsi = 100 - (100/denum);
    
    return rsi;
    
}

double M_t(vector<double> &y, double n )
{
    double m = 0;
    vector<double> dum;
    
    for (double i = 0; i< y.size() - n; i++) {
        dum.push_back(y.at(i));
    }
    double H_t = ret_max(dum);
    double L_t = ret_min(dum);
    double p_t = dum.at(dum.size() - 1);
    
    m = (H_t + L_t + p_t)/3;
    
    return m;
}

double CCI (vector<double> &y , double n)
{
    double SM_t =0;
    
    for (double j =0 ; j < n ; j++) {
        SM_t += M_t(y, j);
        
    }
    
    SM_t = SM_t/n;
    double D_t = 0;
    
    for (double i = 0; i<n; i++) {
        D_t += abs(M_t(y, i) - SM_t);
    }
    
    double M_dum = M_t(y, 0.0);
    
    double numer = M_dum - SM_t;
    
   double c = numer/ (0.015*D_t);
    
    return c;
    
}

vector<double> starting_point (vector<double> &t, int n)
{
    vector<double> y ;
    
    for (int i = 0; i<= n ; ++i) {
        y.push_back(t.at(i));
    }
    
    return y;
}

int main(int argc,   char * argv[]) {
    
    vector <unsigned> frame;
    vector<unsigned> lattice;
    vector<unsigned> org;
    vector<unsigned> stack;
    
    frame.push_back(10);
    frame.push_back(20);// increase number of hidden nerous, then perhaps try adding one more hidden layer
    frame.push_back(1);
    
    lattice.push_back(10);
    lattice.push_back(11);// increase number of hidden nerous, then perhaps try adding one more hidden layer
    lattice.push_back(1);
    
    org.push_back(10);
    org.push_back(11); // increase number of hidden nerous, then perhaps try adding one more hidden layer
    org.push_back(1);
    
    stack.push_back(10);
    stack.push_back(11);// increase number of hidden nerous, then perhaps try adding one more hidden layer
    stack.push_back(1);
    
    
    //declare different sample
    Net nn(frame);
    
    vector <double> avg_nn;
    
    unsigned int counter = 0;
    unsigned int count = 1;
    
    ifstream myReadFile;
    myReadFile.open("goog_05_17_input_train_p.csv");
    
    ifstream myTargetVal;
    myTargetVal.open("Google_Ouptut_Train_Sign.csv");
    //myTargetVal.open("goog_05_17_return_output_train_p.csv");
    
    double s;
    double t;
    
    if(myReadFile.is_open()) cout << "File is open" << endl;
    
    vector<double> target_Vals;
    vector<double> dummy_input;
    vector <double> resultVals;
    
    vector<double> prior_weights;
    ifstream recursive_weight_file;
    
    recursive_weight_file.open("Wrong_w_c.csv");
    if(recursive_weight_file.is_open()) cout << "File is open" << endl;
    
    //if(pastWeightFile.is_open()) {cout << "File is open" << endl;} else {cout << "Not open";}
    double w;
    while (!recursive_weight_file.eof()) {
        
        recursive_weight_file >> w;
        prior_weights.push_back(w);
        
    }
    
    //cout <<prior_weights.at(1) << endl;
    
    // Collect all training data points
    while (!myReadFile.eof()) {
        //Need to skip next point as next input to avoid misguidance.
        myReadFile >> s;
        dummy_input.push_back(s);
        //cout << "Size of dummy_input: " << dummy_input.size() << endl;
        myTargetVal >> t;
        target_Vals.push_back(t);
    }
    
    //for (int j = 14; j<target_Vals.size(); j++) {
      //  cout << target_Vals.at(j) << endl;
    //}
    //return 9;
    nn.read_past_weights(prior_weights); //take prior weights from already prior weighted NN
    
   //get starting data to get the indicators rolling
    vector<double> start_vect(starting_point(dummy_input, 14));
    
    cout << "indicators: " << endl ;
    //call indicators with start_vect
    double moving_avg = moving_average(start_vect, 14);
    double compare = moving_average(start_vect, 14);
    double weight_avg = weight_moving_avg(start_vect, 14);
    double moment = momentum(start_vect, 14);
    double stocast_k = stochastic_k(start_vect, 14);
    double stocast_d = stochastic_d(start_vect, 14);
    double psi = RSI(start_vect, 14);
    double lary = larry_william(start_vect, 14);
    double ad = accum_dist(start_vect, 14);
    double cci = CCI(start_vect, 14);
    
    cout <<"move average: " <<  moving_avg << endl;
    cout << "compare vect: " << compare << endl;
    
    //return 8;
    
    
    vector<double> tech_inds;
    tech_inds.push_back(moving_avg);
    tech_inds.push_back(weight_avg);
    tech_inds.push_back(moment);
    tech_inds.push_back(stocast_k);
    tech_inds.push_back(stocast_d);
    tech_inds.push_back(psi);
    tech_inds.push_back(lary);
    tech_inds.push_back(ad);
    tech_inds.push_back(cci);
    //tech_inds.push_back(moving_avg);
    
    
    
    //check to see if indicators are giving me values
    cout << "moving average: " << moving_avg << endl;
    cout << "weighted average: " << weight_avg << endl;
    cout << "momentum: " << moment << endl;
    cout << "stochastic k: " << stocast_k << endl;
    cout << "stochastic d: " << stocast_d << endl;
    cout << "RSI: " << psi << endl;
    cout << "William Larry: " << lary << endl;
    cout << "accumulated distribution: " << ad << endl;
    cout << "CCI: " << cci << endl;
    
    //while (!myReadFile.eof()) {
        //Need to skip next point as next input to avoid misguidance.
    //vector<double> dum_put(tech_inds);
    vector<double> out_put;
    vector<double> get_results;
    int cont = dummy_input.size();
    
   // for(int it = 0; it < 30; it++) { //for number of training cycles, code is too slow hence one cycle performed
        for (int i = 14; i < target_Vals.size(); i++){
        //myReadFile >> s;
            //cout << "i: " << i << endl;
            tech_inds.clear(); //CLEAR TECH INDICATORS TO MAKE SURE INPUT MATCH NN
        
            
            tech_inds.push_back(moving_avg);
            tech_inds.push_back(weight_avg);
            tech_inds.push_back(moment);
            tech_inds.push_back(stocast_k);
            tech_inds.push_back(stocast_d);
            tech_inds.push_back(psi);
            tech_inds.push_back(lary);
            tech_inds.push_back(ad);
            tech_inds.push_back(cci);
            
            //vector<double> dum_put(tech_inds);
            //dum_put.push_back(dummy_input.at(i))
            tech_inds.push_back(dummy_input.at(i));
        //dummy_input.push_back(s);
        //cout << "Size of dummy_input: " << dummy_input.size() << endl;
        //myTargetVal >> t;
        out_put.push_back(target_Vals.at(i));
            //cout <<"size of input: " << dum_put.size() << endl;
        
        nn.feedForward(tech_inds); //net 1
        nn.getResults(get_results);
        nn.backProp(out_put); //Net 1
        
            start_vect.push_back(dummy_input.at(i));
            moving_avg = moving_average(start_vect, 14);
            weight_avg = weight_moving_avg(start_vect, 14);
            moment = momentum(start_vect, 14);
            stocast_k = stochastic_k(start_vect, 14);
            stocast_d = stochastic_d(start_vect, 14);
            psi = RSI(start_vect, 14);  //FIX THIS
            lary = larry_william(start_vect, 14);
            ad = accum_dist(start_vect, 14);
            cci = CCI(start_vect, 14);
            
            //dum_put.clear();
        out_put.clear();
        //target_Vals.clear();
    //} //End of while loop
            cont =i;
    
    }
       // cout <<"current iterr: " << it;
//} //end of iteration
    
    //PERFORM OUT OF SAMPLE TEST FOR GRANDCHILD NETWORK
    cout << "cont is: " <<  cont << endl;
    cout <<" size of start vect: " << start_vect.size() << endl;
    //cout << "return 9: " << endl;
    //return 9;
    
    ifstream OFST_input_file;
    ifstream OFST_output_file;
    OFST_input_file.open("goog_input_18_test_p.csv");
    OFST_output_file.open("Google_test_out_Sign.csv");
    //OFST_output_file.open("googl_test_18_output_ret_p.csv");
    
    // To get the results of the different NNs to later average them .
    vector<double> OFST_results;
   
    
    // VECTORS TO COLLECT AND COMPARE  OUTPUTS
    
    vector<double> collect;
    
    cout << "OUT OF SAMPLE TESTING" << endl;
    if (OFST_input_file.is_open() && OFST_output_file.is_open()) {
        cout << "Files open "<< endl;
    } else {cout<< "Not open" << endl; }
    
    double a =0;
  
    tech_inds.clear();
    
    
    while (!OFST_output_file.eof()) {
        
        //double sum = 0;
        tech_inds.clear();
        
        OFST_input_file >> s;
        //dummy_input.push_back(s);
        
        tech_inds.push_back(moving_avg);
        tech_inds.push_back(weight_avg);
        tech_inds.push_back(moment);
        tech_inds.push_back(stocast_k);
        tech_inds.push_back(stocast_d);
        tech_inds.push_back(psi);
        tech_inds.push_back(lary);
        tech_inds.push_back(ad);
        tech_inds.push_back(cci);
        tech_inds.push_back(s);
        
        //cout << "Size of dummy_input: " << dummy_input.size() << endl;
        
        OFST_output_file >> t;
        target_Vals.push_back(t); //put in target values
        out_put.push_back(t);
        
        
        nn.feedForward(tech_inds); //net 1
        nn.getResults(OFST_results);
        cout << OFST_results.back() << endl;

        a = OFST_results.back();
        
        collect.push_back(a);

        //Back propagate all networks called
        nn.backProp(out_put); //Net 1

        start_vect.push_back(s);
        
        moving_avg = moving_average(start_vect, 14);
        weight_avg = weight_moving_avg(start_vect, 14);
        moment = momentum(start_vect, 14);
        stocast_k = stochastic_k(start_vect, 14);
        stocast_d = stochastic_d(start_vect, 14);
        psi = RSI(start_vect, 14);  //FIX THIS
        lary = larry_william(start_vect, 14);
        ad = accum_dist(start_vect, 14);
        cci = CCI(start_vect, 14);
        
        
        //dummy_input.clear();
        //target_Vals.clear();
        OFST_results.clear();
        out_put.clear();
        
        counter = ++count;
    }
    
    return 0;
}


