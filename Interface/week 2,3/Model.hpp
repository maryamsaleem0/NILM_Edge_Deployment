#ifndef MODEL_HPP
#define MODEL_HPP
#include <cstdint>

class ModelListener;

class Model
{
public:
    Model();

    void bind(ModelListener* listener)
    {
        modelListener = listener;
    }

    void tick();
    void setSelectedAppliance(uint8_t id);
    uint8_t getSelectedAppliance();
protected:
    ModelListener* modelListener;
    uint8_t selectedApplianceID;
};

#endif // MODEL_HPP
