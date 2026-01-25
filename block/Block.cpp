#include <vector>

struct Concept {
    virtual void apply() = 0;
    virtual ~Concept() = default;
};

template<typename T>
struct Model : public Concept { // тут надо подумать 
public:
    Model(const T &data) : data_(data) {
    }

    
    
private:
    T data_;
};


class Block {
    Block() = default;

};

class NeuralNetwork {
    public:
        void addLayer(Block block) {
            blocks.push_back(block);
        }
    private:
        std::vector<Block> blocks;
};

class LinearLayer {

};

class SigmoidLayer {

};

int main() {
    NeuralNetwork nn;
    nn.addLayer(LinearLayer());
    nn.addLayer(SigmoidLayer());
    nn.addLayer(LinearLayer());
    nn.addLayer(SigmoidLayer());
}
