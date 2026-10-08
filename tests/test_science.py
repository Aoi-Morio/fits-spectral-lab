from io import BytesIO

import numpy as np
import pytest
from astropy.io import fits

from spectral_lab.io import Spectrum, read_fits, read_csv, export_fits, export_csv, image_axis, demo_spectrum
from spectral_lab.analysis import continuum_fit, fit_line


def image_bytes(data, header=None):
    buf = BytesIO()
    fits.PrimaryHDU(data, header=header).writeto(buf)
    return buf.getvalue()


@pytest.mark.parametrize("amplitude,kind", [(15.0, "emission"), (-6.0, "absorption")])
def test_known_line_center_width_and_integrated_flux(amplitude, kind):
    x = np.linspace(6500, 6600, 1001)
    continuum = 10 + 0.01 * (x - 6550)
    sigma = 1.8
    y = continuum + amplitude * np.exp(-0.5 * ((x - 6562.8) / sigma)**2)
    spectrum = Spectrum(x, y, np.full(x.size, 0.1), "Angstrom")
    estimated, mask = continuum_fit(spectrum, excluded=[(6552, 6574)])
    assert np.max(np.abs(estimated - continuum)) < 0.001
    result, _, _ = fit_line(spectrum, estimated, 6552, 6574, kind)
    assert result.center == pytest.approx(6562.8, abs=0.001)
    assert result.fwhm == pytest.approx(2.354820045 * sigma, rel=0.001)
    assert result.gaussian_flux == pytest.approx(amplitude * sigma * np.sqrt(2*np.pi), rel=0.001)
    assert np.sign(result.equivalent_width) == -np.sign(amplitude)
    assert result.center_error is not None and result.center_error > 0
    assert not mask[(x >= 6552) & (x <= 6574)].any()


def test_fit_is_stable_for_physical_small_flux_units():
    spectrum = demo_spectrum()
    spectrum.flux *= 1e-17
    spectrum.error *= 1e-17
    continuum, _ = continuum_fit(spectrum, excluded=[(6555, 6571)])
    result, _, _ = fit_line(spectrum, continuum, 6555, 6571)
    assert result.center == pytest.approx(6562.8, abs=0.1)
    assert result.gaussian_flux == pytest.approx(24 * 1.7 * np.sqrt(2*np.pi) * 1e-17, rel=0.05)


def test_linear_wcs_has_correct_one_based_reference_and_nm_conversion():
    header = fits.Header(dict(CTYPE1="WAVE", CRVAL1=500.0, CRPIX1=3.0, CDELT1=0.2, CUNIT1="nm"))
    spectrum = read_fits(image_bytes(np.ones(20), header))
    assert spectrum.x[2] == pytest.approx(5000)
    assert spectrum.x[0] == pytest.approx(4996)
    assert spectrum.x_unit == "Angstrom"


def test_non_linear_log_wcs_is_interpreted_with_astropy():
    header = fits.Header(dict(CTYPE1="WAVE-LOG", CRVAL1=5000.0, CRPIX1=1.0, CDELT1=10.0, CUNIT1="Angstrom"))
    spectrum = read_fits(image_bytes(np.ones(20), header))
    assert spectrum.x[5] == pytest.approx(5000 * np.exp(50 / 5000))


def test_sdss_legacy_log_wavelength():
    spectrum = read_fits(image_bytes(np.ones(20), fits.Header(dict(COEFF0=3.5, COEFF1=0.001))))
    assert spectrum.x[5] == pytest.approx(10**3.505)


def test_unknown_calibration_stays_pixels():
    spectrum = read_fits(image_bytes(np.ones(20)))
    assert spectrum.x_unit == "pixel"
    np.testing.assert_array_equal(spectrum.x, np.arange(20))


def test_2d_aperture_background_subtraction_and_overlap_rejection():
    image = np.ones((6, 20)) * 7
    image[2:4] += np.arange(20)
    raw = image_bytes(image)
    spectrum = read_fits(raw, aperture=(2, 4), background=(0, 2))
    np.testing.assert_allclose(spectrum.flux, 2 * np.arange(20))
    with pytest.raises(ValueError, match="重ならない"):
        read_fits(raw, aperture=(2, 4), background=(1, 3))


def test_vertical_dispersion_uses_axis2_wcs():
    header = fits.Header(dict(CTYPE1="LINEAR", CTYPE2="WAVE", CRPIX1=1, CRPIX2=1,
                             CRVAL1=0, CRVAL2=650, CDELT1=1, CDELT2=0.1, CUNIT2="nm"))
    raw = image_bytes(np.ones((20, 5)), header)
    spectrum = read_fits(raw, dispersion_axis=2, aperture=(1, 4))
    np.testing.assert_allclose(spectrum.flux, 3)
    assert spectrum.x[10] == pytest.approx(6510)


@pytest.mark.parametrize("unit", ["Angstrom", "pixel"])
def test_fits_export_roundtrip_preserves_samples_and_errors(unit):
    source = demo_spectrum()
    if unit == "pixel":
        source.x = np.arange(source.x.size, dtype=float)
        source.x_unit = "pixel"
    loaded = read_fits(export_fits(source), 1)
    np.testing.assert_allclose(loaded.x, source.x)
    np.testing.assert_allclose(loaded.flux, source.flux)
    np.testing.assert_allclose(loaded.error, source.error)
    assert loaded.x_unit == source.x_unit


def test_sdss_table_inverse_variance_and_bad_pixel_filter():
    x = np.linspace(3.6, 3.7, 20)
    ivar = np.full(20, 4.)
    ivar[3] = 0
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="loglam", format="D", array=x),
        fits.Column(name="flux", format="D", array=np.ones(20)),
        fits.Column(name="ivar", format="D", array=ivar),
    ])
    buffer = BytesIO()
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(buffer)
    spectrum = read_fits(buffer.getvalue(), 1)
    assert spectrum.x.size == 19
    np.testing.assert_allclose(spectrum.error, 0.5)
    assert spectrum.x[0] == pytest.approx(10**3.6)


def test_csv_reverse_axis_roundtrip_and_invalid_errors():
    raw = "wavelength,flux,error\n" + "\n".join(f"{500 + i},{i+1},0.1" for i in reversed(range(20)))
    spectrum = read_csv(raw.encode(), "nm")
    assert spectrum.x[0] == pytest.approx(5000)
    loaded = read_csv(export_csv(spectrum).encode())
    np.testing.assert_allclose(loaded.flux, spectrum.flux)
    with pytest.raises(ValueError, match="8点"):
        read_csv(("\n".join(f"{i},1,-1" for i in range(20))).encode())


def test_duplicate_axis_and_cube_fail_explicitly():
    with pytest.raises(ValueError, match="重複"):
        Spectrum(np.ones(10), np.ones(10))
    with pytest.raises(ValueError, match="キューブ"):
        read_fits(image_bytes(np.ones((4, 4, 20))))


@pytest.mark.parametrize("with_error", [True, False])
def test_processed_csv_can_be_reopened_without_using_continuum_as_error(with_error):
    original = demo_spectrum()
    if not with_error:
        original.error = None
    continuum, _ = continuum_fit(original, excluded=[(6555, 6571)])
    reopened = read_csv(export_csv(original, continuum).encode())
    np.testing.assert_allclose(reopened.flux, original.flux)
    if with_error:
        np.testing.assert_allclose(reopened.error, original.error)
    else:
        assert reopened.error is None


def test_nonpositive_continuum_has_no_equivalent_width():
    x = np.linspace(0, 20, 201)
    continuum = np.zeros(201)
    flux = np.exp(-0.5*((x-10)/1.5)**2)
    result, _, _ = fit_line(Spectrum(x, flux), continuum, 3, 17)
    assert result.equivalent_width is None


def test_coupled_spectral_wcs_is_rejected_instead_of_fake_calibration():
    header = fits.Header(dict(NAXIS=2, CTYPE1="WAVE", CTYPE2="LINEAR", CUNIT1="nm",
                             CRPIX1=1, CRPIX2=1, CRVAL1=500, CRVAL2=0,
                             CD1_1=1, CD1_2=0.1, CD2_1=0, CD2_2=1))
    with pytest.raises(ValueError, match="結合"):
        image_axis(header, 20)
