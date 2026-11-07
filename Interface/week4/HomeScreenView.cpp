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
    // AC  ID 0
    presenter->setSelectedAppliance(0); 
}

void HomeScreenView::fanDetailsClicked()
{
    // Fan  ID 1 
    presenter->setSelectedAppliance(1); 
}

void HomeScreenView::lightDetailsClicked()
{
    // Light ID 2 
    presenter->setSelectedAppliance(2); 
}

void HomeScreenView::fridgeDetailsClicked()
{
    // Fridge  ID 3 
    presenter->setSelectedAppliance(3); 
}