#include <gui/model/Model.hpp>
#include <gui/model/ModelListener.hpp>
#include <cmath>    // Sine wave ke liye
#include <cstdlib>  // Random numbers ke liye
Model::Model() : modelListener(0)
{

}

// void Model::tick()
// {
// static int tickCounter = 0;
//     tickCounter++;

//     // Har 5th frame par naya data point banayein
//     if (tickCounter % 5 == 0)
//     {
//         // Fake data: 1000W base + sine wave + random noise[cite: 1]
//         int fakePower = 1000 + (500 * sin(tickCounter * 0.1)) + (rand() % 100);

//         // Agar listener (Presenter) maujood hai toh data aage bhej dein[cite: 1]
//         if (modelListener != 0)
//         {
//             modelListener->updatePowerGraph(fakePower);
//         }
//     }
// }
void Model::tick()
{
    static int tickCounter = 0;
    tickCounter++;

    // Har 5th frame par naya data point banayein
    if (tickCounter % 5 == 0)
    {
        // Graph ka fake data
        int fakePower = 1000 + (500 * sin(tickCounter * 0.1)) + (rand() % 100);

        // 4 Appliances ka fake data (Watts mein)
        int fakeAc = 500 + (rand() % 50);
        int fakeFan = 80 + (rand() % 10);
        int fakeLight = 40 + (rand() % 5);
        int fakeFridge = 150 + (rand() % 20);

        // Data Presenter ko bhej dein
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