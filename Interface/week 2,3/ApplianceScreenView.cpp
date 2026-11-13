#include <gui/appliancescreen_screen/ApplianceScreenView.hpp>
#include <touchgfx/Color.hpp>


ApplianceScreenView::ApplianceScreenView()
{

}

void ApplianceScreenView::tearDownScreen()
{
    ApplianceScreenViewBase::tearDownScreen();
}

void ApplianceScreenView::setupScreen()
{
    ApplianceScreenViewBase::setupScreen();

    // Presenter se ID mangwayein
    uint8_t applianceId = presenter->getSelectedAppliance();
    int currentPower = 0; // Mock data

    // 1. ID ke hisaab se Appliance ka naam aur Power set karein
    switch(applianceId)
    {
        case 0: // AC
            Unicode::snprintf(textArea2Buffer, TEXTAREA2_SIZE, "AC");
            currentPower = 1500; 
            break;
        case 1: // Fan
            Unicode::snprintf(textArea2Buffer, TEXTAREA2_SIZE, "Fan");
            currentPower = 80;
            break;
        case 2: // Light
            Unicode::snprintf(textArea2Buffer, TEXTAREA2_SIZE, "Light");
            currentPower = 0; // OFF check karne ke liye 0 rakha hai
            break;
        case 3: // Fridge
            Unicode::snprintf(textArea2Buffer, TEXTAREA2_SIZE, "Fridge");
            currentPower = 250;
            break;
    }

    // 2. Power ka text aur Progress Bar update karein
    Unicode::snprintf(textArea3Buffer, TEXTAREA3_SIZE, "%d", currentPower);
    boxProgress1.setValue(currentPower);

    // 3. Status ON/OFF aur Circle ka Color
    if (currentPower == 0)
    {
        // OFF State - Red Circle aur OFF text
        Unicode::snprintf(textArea4Buffer, TEXTAREA4_SIZE, "OFF");
        circle1Painter.setColor(touchgfx::Color::getColorFromRGB(255, 0, 0));
    }
    else
    {
        // ON State - Green Circle aur ON text
        Unicode::snprintf(textArea4Buffer, TEXTAREA4_SIZE, "ON");
        circle1Painter.setColor(touchgfx::Color::getColorFromRGB(0, 255, 0));
    }

    // 4. Sab update karne ke baad screen ko refresh (invalidate) karein
    textArea2.invalidate();
    textArea3.invalidate();
    textArea4.invalidate();
    boxProgress1.invalidate();
    circle1.invalidate();
}
