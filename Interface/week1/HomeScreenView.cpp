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