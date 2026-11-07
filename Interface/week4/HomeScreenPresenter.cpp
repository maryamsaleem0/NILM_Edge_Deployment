#include <gui/homescreen_screen/HomeScreenView.hpp>
#include <gui/homescreen_screen/HomeScreenPresenter.hpp>

HomeScreenPresenter::HomeScreenPresenter(HomeScreenView& v)
    : view(v)
{

}

void HomeScreenPresenter::activate()
{

}

void HomeScreenPresenter::deactivate()
{

}
void HomeScreenPresenter::updatePowerGraph(int powerValue)
{
    // send Data  to View (Screen) 
    view.addGraphData(powerValue);
}
void HomeScreenPresenter::updateAppliances(int ac, int fan, int light, int fridge)
{
    view.updateApplianceUI(ac, fan, light, fridge);
}
void HomeScreenPresenter::setSelectedAppliance(uint8_t id)
{
    // Presenter sent this ID to Model (Brain) 
    model->setSelectedAppliance(id);
}