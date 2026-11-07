#include <gui/model/Model.hpp>
#include <gui/model/ModelListener.hpp>
Model::Model() : modelListener(0)
{

}
void Model::tick()
{
    static int tickCounter = 0;
    static int safe_random = 0; // Hardware-safe random number generator

    tickCounter++;

    // at each 5th frame new data point form
    if (tickCounter % 5 == 0)
    {
        safe_random = (safe_random + 27) % 500; 

        int fakePower = 1000 + safe_random;

        // 4 Appliances ka fake data
        int fakeAc = 500 + (safe_random % 50);
        int fakeFan = 80 + (safe_random % 10);
        int fakeLight = 40 + (safe_random % 5);
        int fakeFridge = 150 + (safe_random % 20);

        if (modelListener != 0)
        {
            modelListener->updatePowerGraph(fakePower);
            modelListener->updateAppliances(fakeAc, fakeFan, fakeLight, fakeFridge);
        }
    }
}

void Model::setSelectedAppliance(uint8_t id)
{
    selectedApplianceID = id;
}

uint8_t Model::getSelectedAppliance()
{
    return selectedApplianceID;
}