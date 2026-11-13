#include <gui/appliancescreen_screen/ApplianceScreenView.hpp>
#include <gui/appliancescreen_screen/ApplianceScreenPresenter.hpp>

ApplianceScreenPresenter::ApplianceScreenPresenter(ApplianceScreenView& v)
    : view(v)
{

}

void ApplianceScreenPresenter::activate()
{

}

void ApplianceScreenPresenter::deactivate()
{

}
uint8_t ApplianceScreenPresenter::getSelectedAppliance()
{
    return model->getSelectedAppliance();
}