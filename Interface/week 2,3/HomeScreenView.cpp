#include <gui/homescreen_screen/HomeScreenView.hpp>

HomeScreenView::HomeScreenView()
{

}

void HomeScreenView::setupScreen()
{
    HomeScreenViewBase::setupScreen();
}

void HomeScreenView::tearDownScreen()
{
    HomeScreenViewBase::tearDownScreen();
}
void HomeScreenView::addGraphData(int powerValue)
{
    // Graph widget mein data point add kar rahe hain
    dynamicGraph1.addDataPoint(powerValue);
}
void HomeScreenView::updateApplianceUI(int ac, int fan, int light, int fridge)
{
    // Har tile ko uski power value bhej rahe hain
    applianceTile1.setPower(ac);
    applianceTile2.setPower(fan);
    applianceTile3.setPower(light);
    applianceTile4.setPower(fridge);
}
void HomeScreenView::acDetailsClicked()
{
    // AC ki ID 0 bhej rahe hain
    presenter->setSelectedAppliance(0); 
}

void HomeScreenView::fanDetailsClicked()
{
    // Fan ki ID 1 bhej rahe hain
    presenter->setSelectedAppliance(1); 
}

void HomeScreenView::lightDetailsClicked()
{
    // Light ki ID 2 bhej rahe hain
    presenter->setSelectedAppliance(2); 
}

void HomeScreenView::fridgeDetailsClicked()
{
    // Fridge ki ID 3 bhej rahe hain
    presenter->setSelectedAppliance(3); 
}