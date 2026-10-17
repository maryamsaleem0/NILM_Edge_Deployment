#include <gui/model/Model.hpp>
#include <gui/model/ModelListener.hpp>
#include <cmath>    // Sine wave ke liye
#include <cstdlib>  // Random numbers ke liye
Model::Model() : modelListener(0)
{

}

void Model::tick()
{
static int tickCounter = 0;
    tickCounter++;

    // Har 5th frame par naya data point banayein
    if (tickCounter % 5 == 0)
    {
        // Fake data: 1000W base + sine wave + random noise[cite: 1]
        int fakePower = 1000 + (500 * sin(tickCounter * 0.1)) + (rand() % 100);

        // Agar listener (Presenter) maujood hai toh data aage bhej dein[cite: 1]
        if (modelListener != 0)
        {
            modelListener->updatePowerGraph(fakePower);
        }
    }
}
