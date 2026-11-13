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
    // Data ko View (Screen) ki taraf bhej rahe hain
    view.addGraphData(powerValue);
}